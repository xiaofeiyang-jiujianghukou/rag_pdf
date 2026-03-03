import os
import sys
from typing import Any, List, Optional

# ================= LangChain 核心导入 (2026 新版) =================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 阿里云原生 SDK (用于底层调用)
import dashscope
from dashscope import Generation

# ================= 配置区域 =================
API_KEY = "sk-a41ebd38c93c43269e46a6ec46787f18"
PDF_FILE = "demo.pdf"
INDEX_DIR = "./faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")

# 模型名称
MODEL_EMBED = "text-embedding-v3"
MODEL_LLM = "qwen-plus"

if API_KEY == "你的_阿里云_DashScope_API_KEY":
    if os.getenv("DASHSCOPE_API_KEY"):
        API_KEY = os.getenv("DASHSCOPE_API_KEY")
    else:
        print("❌ 请填入 API_KEY"); sys.exit(1)

dashscope.api_key = API_KEY

# ================= 自定义 DashScope 组件 =================
# 因为官方 langchain-dashscope 可能更新不及时，我们手动实现最稳的 Wrapper

class DashScopeEmbeddings(Embeddings):
    def __init__(self, model=MODEL_EMBED):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量"""
        if not texts:
            return []
        
        results = []
        batch_size = 20
        
        # 分批次请求，避免超时或长度限制
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # 调用阿里云 API
            resp = dashscope.TextEmbedding.call(model=self.model, input=batch)
            
            if resp.status_code != 200:
                raise Exception(f"DashScope Embedding 错误: {resp.code} - {resp.message}")
            
            # 解析结果
            # 注意：API 返回结构通常是 {'embeddings': [{'embedding': [...], 'index': 0}, ...]}
            # 不需要也不应该去取 'text' 字段，直接按列表顺序取即可
            try:
                batch_embeddings = [item['embedding'] for item in resp.output['embeddings']]
                results.extend(batch_embeddings)
            except KeyError as e:
                raise Exception(f"API 返回数据结构 unexpected: {e}. 原始返回: {resp.output}")
                
        return results

    def embed_query(self, text: str) -> List[float]:
        """生成单个查询向量"""
        return self.embed_documents([text])[0]

class DashScopeLLM(LLM):
    # ... (LLM 类保持不变)
    @property
    def _llm_type(self) -> str:
        return "dashscope"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            resp = Generation.call(
                model=MODEL_LLM,
                prompt=prompt,
                api_key=API_KEY,
                result_format='message'
            )
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            else:
                return f"Error: {resp.message}"
        except Exception as e:
            return f"Exception: {str(e)}"

# ================= 主流程 =================

print("⚙️ 初始化 LangChain 组件...")
embeddings = DashScopeEmbeddings(model=MODEL_EMBED)
llm = DashScopeLLM()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 1. 加载或创建向量库
if os.path.exists(INDEX_PATH):
    print(f"📂 加载现有索引: {INDEX_PATH}")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"📄 处理新文档: {PDF_FILE}")
    if not os.path.exists(PDF_FILE):
        print(f"❌ 文件不存在: {PDF_FILE}"); sys.exit(1)
    
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    print(f"   读取了 {len(docs)} 页文档")
    
    splits = text_splitter.split_documents(docs)
    print(f"   切分为 {len(splits)} 个片段")
    
    if not splits:
        print("❌ 未提取到文本，可能是扫描版 PDF"); sys.exit(1)
        
    print("   🔢 正在生成向量 (这可能需要一点时间)...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ 索引已保存至 {INDEX_DIR}")

# 2. 构建检索链 (RAG Chain)
# 使用 LCEL 语法构建，这是 LangChain 推荐的现代写法
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """基于以下已知信息，简洁和专业地回答用户的问题。
如果无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许编造答案。

已知信息：
{context}

问题：{question}
回答："""

prompt = ChatPromptTemplate.from_template(template)

# 定义如何处理上下文
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 组装链条
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n🤖 系统就绪！(输入 quit 退出)")

while True:
    try:
        query = input("\n你: ").strip()
        if not query: continue
        if query.lower() in ['quit', 'exit']: break
        
        print("🤖 思考中...", end='\r')
        response = rag_chain.invoke(query)
        print(" " * 20, end='\r') # 清除提示
        print(f"🤖 AI: {response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()