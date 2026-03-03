import os
import sys
import json
import requests
import numpy as np

# 【关键】强制设置标准输出为 UTF-8，防止控制台打印中文报错
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings

# ================= 自定义 Embeddings 类 (使用原生 requests) =================
class SafeDashScopeEmbeddings(Embeddings):
    def __init__(self, api_key, model="text-embedding-v3"):
        self.api_key = api_key
        self.model = model
        # 阿里云兼容模式地址
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts):
        """
        批量生成向量。
        严格确保输入是 List[str]，并处理空值。
        """
        # 1. 数据清洗：确保所有元素都是字符串，且非空
        clean_texts = []
        valid_indices = []
        
        for i, t in enumerate(texts):
            if isinstance(t, str) and t.strip():
                clean_texts.append(t.strip())
                valid_indices.append(i)
            else:
                print(f"⚠️ 跳过第 {i} 个无效片段 (类型: {type(t)}, 内容: {t})")
        
        if not clean_texts:
            raise ValueError("没有有效的文本片段可以生成向量！")

        # 2. 分批发送 (避免单次请求过大)
        batch_size = 20
        all_embeddings = [None] * len(texts) # 预分配结果列表
        
        print(f"   -> 开始处理 {len(clean_texts)} 个有效片段...")
        
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i : i + batch_size]
            
            # 构造符合阿里云兼容模式的 Payload
            # 注意：input 必须是纯粹的 string list
            payload = {
                "model": self.model,
                "input": batch, 
                "encoding_format": "float"
            }
            
            try:
                # 使用 requests 发送，json 参数会自动处理 UTF-8 编码
                response = requests.post(
                    self.url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"API 错误 {response.status_code}: {response.text}")
                
                result = response.json()
                
                # 解析返回结果
                # 阿里云返回格式: { "data": [ { "index": 0, "embedding": [...] }, ... ] }
                batch_vectors_map = {item["index"]: item["embedding"] for item in result["data"]}
                
                # 将结果填回总列表
                for idx_in_batch, vector in batch_vectors_map.items():
                    original_idx = valid_indices[i + idx_in_batch]
                    all_embeddings[original_idx] = vector
                    
                print(f"   -> 批次完成: {i//batch_size + 1}/{(len(clean_texts)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"❌ 请求失败: {e}")
                raise e
        
        # 检查是否有遗漏 (理论上不会)
        if None in all_embeddings:
            raise RuntimeError("部分向量生成失败，结果不完整。")
            
        return all_embeddings

    def embed_query(self, text):
        """单个查询生成向量"""
        return self.embed_documents([text])[0]

# ================= 主程序 =================

# 1. 配置 API
DASHSCOPE_API_KEY = "sk-a41ebd38c93c43269e46a6ec46787f18"  # <--- 请填入 Key

if DASHSCOPE_API_KEY == "你的_阿里云_DashScope_API_KEY":
    print("❌ 请先在代码中填入你的 DASHSCOPE_API_KEY！")
    exit()

print("⚙️ 正在初始化 Qwen-Plus + 自定义 Embeddings 环境...")

# 2. 加载 PDF
print("📄 正在加载 PDF...")
try:
    file_path = "demo.pdf"
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}, 当前目录: {os.getcwd()}")
        exit()
        
    loader = PyPDFLoader(file_path) 
    docs = loader.load()
    print(f"✅ 加载完成，共 {len(docs)} 页。")
except Exception as e:
    print(f"❌ 加载 PDF 失败: {e}")
    exit()

# 3. 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"✂️ 切分完成，共 {len(splits)} 个片段。")

if not splits:
    print("⚠️ 未切分出任何内容。")
    exit()

# 🔥 清洗元数据 (防止 FAISS 保存时报错)
print("🧹 正在清洗元数据...")
for i, doc in enumerate(splits):
    new_meta = {}
    for k, v in doc.metadata.items():
        if isinstance(v, str):
            try:
                v.encode('ascii')
                new_meta[k] = v
            except UnicodeEncodeError:
                # 遇到中文路径或属性，替换为安全字符串
                new_meta[k] = f"safe_{k}_{i}"
        else:
            new_meta[k] = v
    doc.metadata = new_meta

# 4. 向量化与存储
print("🔢 正在生成向量 (使用 text-embedding-v3)...")

# 实例化自定义 Embeddings
embeddings = SafeDashScopeEmbeddings(api_key=DASHSCOPE_API_KEY, model="text-embedding-v3")

try:
    # 提取文本和内容
    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]
    
    # 使用 LangChain 的 from_texts，它会调用我们自定义的 embed_documents
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    save_path = "./faiss_index_qwen"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    vectorstore.save_local(save_path)
    print(f"✅ FAISS 向量库建立成功！保存至: {save_path}")

except Exception as e:
    print(f"❌ 向量库构建失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 5. 设置检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 设置大模型
llm = ChatOpenAI(
    model="qwen-plus", 
    temperature=0.7,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY
)

# 7. 提示词模板
prompt_template = """你是一个智能助手。请严格根据提供的【上下文】回答用户问题。
如果上下文中没有答案，请直接说“根据提供的文档，我无法找到答案”。

【上下文】:
{context}

【问题】:
{question}

【回答】:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 8. 构建 Chain
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    use_legacy_chain = True
except ImportError:
    setup_context = {"context": retriever, "question": RunnablePassthrough()}
    rag_chain = setup_context | prompt | llm
    use_legacy_chain = False

# 9. 开始交互
print("\n🤖 Qwen-RAG (原生请求版) 已就绪！(输入 'quit' 退出)")

while True:
    try:
        query = input("\n你: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query.strip(): continue

        if use_legacy_chain:
            response = rag_chain.invoke({"input": query})
            answer = response.get("answer", "无回答")
        else:
            response = rag_chain.invoke(query)
            answer = response.content
        
        print(f"🤖 Qwen: {answer}")
    except Exception as e:
        print(f"❌ 问答错误: {e}")