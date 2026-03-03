import os
import sys

# 环境设置
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ================= 配置区域 =================
API_KEY = "sk-a41ebd38c93c43269e46a6ec46787f18" 
PDF_FILE = "demo.pdf"
INDEX_PATH = "./faiss_index"

if API_KEY == "你的_阿里云_DashScope_API_KEY":
    print("❌ 请填入 API Key"); sys.exit(1)

print("⚙️ 初始化中...")

# 🔥 尝试多种导入路径，自动适配你的环境
try:
    # 尝试 1: 新版社区集成 (推荐)
    from langchain_community.embeddings import DashScopeEmbeddings
    print("✅ 使用 langchain_community.embeddings.DashScopeEmbeddings")
except ImportError:
    try:
        # 尝试 2: 旧版 alibaba 包
        from langchain_alibaba.embeddings import DashScopeEmbeddings
        print("✅ 使用 langchain_alibaba.embeddings.DashScopeEmbeddings")
    except ImportError:
        # 尝试 3: 实在不行，使用最简单的原生封装 (仅 5 行代码，非自定义长类)
        print("⚠️ 未找到现成 Embeddings 类，启用极简原生适配模式...")
        from langchain_core.embeddings import Embeddings
        import dashscope
        
        class DashScopeEmbeddings(Embeddings):
            def __init__(self, model="text-embedding-v3", api_key=None):
                self.model = model
                dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            def embed_documents(self, texts):
                resp = dashscope.TextEmbedding.call(model=self.model, input=texts)
                if resp.status_code != 200: raise Exception(resp.message)
                return [item['embedding'] for item in resp.output['embeddings']]
            def embed_query(self, text):
                return self.embed_documents([text])[0]
        
        print("✅ 已动态创建 DashScopeEmbeddings 类")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 设置 LLM 环境
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 1. 加载与切分
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 🧹 极简清洗 (防止元数据编码报错)
final_docs = []
for doc in splits:
    if not doc.page_content.strip(): continue
    # 将元数据中的非 ASCII 字符移除，防止 FAISS 保存报错
    doc.metadata = {k: (v.encode('ascii', 'ignore').decode() if isinstance(v, str) else v) for k, v in doc.metadata.items()}
    final_docs.append(doc)

print(f"📄 有效片段: {len(final_docs)}")
if not final_docs: print("❌ 无内容"); sys.exit(1)

# 2. 向量化 (自动使用上面导入或创建的类)
print("🔢 生成向量中...")
try:
    # 注意：如果是动态创建的类，需要传入 api_key
    embeddings = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=API_KEY) 
    # 如果上面的类是动态创建的，它可能不接受 dashscope_api_key 参数，而是通过构造函数处理
    # 修正动态类的调用方式：
    if 'dashscope_api_key' in DashScopeEmbeddings.__init__.__code__.co_varnames:
         embeddings = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=API_KEY)
    else:
         embeddings = DashScopeEmbeddings(model="text-embedding-v3", api_key=API_KEY)
         
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ 成功保存至: {INDEX_PATH}")
except Exception as e:
    print(f"❌ 向量化失败: {e}")
    sys.exit(1)

# 3. RAG 链
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="qwen-plus", temperature=0.7)
prompt = ChatPromptTemplate.from_template("根据上下文回答:\n{context}\n问题: {question}\n回答:")
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

print("\n🤖 就绪 (输入 quit 退出)")
while True:
    q = input("\n你: ")
    if q.lower() in ["quit", "exit"]: break
    if not q.strip(): continue
    try:
        print(f"🤖 AI: {chain.invoke({'input': q})['answer']}")
    except Exception as e:
        print(f"❌ 错误: {e}")