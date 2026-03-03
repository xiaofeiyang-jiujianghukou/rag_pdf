import os
import sys
import pickle
import numpy as np

# ================= 配置区域 =================
API_KEY = "sk-a41ebd38c93c43269e46a6ec46787f18"  
PDF_FILE = "demo.pdf"

# 🔥 修改点：指定子目录
INDEX_DIR = "./faiss_index" 
# 确保目录存在
os.makedirs(INDEX_DIR, exist_ok=True)

# 完整的文件路径 (不带后缀，函数内部会加 .faiss 和 .data)
INDEX_PATH = os.path.join(INDEX_DIR, "index_data")

MODEL_EMBED = "text-embedding-v3"
MODEL_LLM = "qwen-plus"

if API_KEY == "你的_阿里云_DashScope_API_KEY":
    env_key = os.getenv("DASHSCOPE_API_KEY")
    if env_key:
        API_KEY = env_key
    else:
        print("❌ 请填入 API_KEY"); sys.exit(1)

print(f"⚙️ 环境就绪 (索引目录: {INDEX_DIR})")

try:
    import dashscope
    import faiss
    from pypdf import PdfReader
except ImportError as e:
    print(f"❌ 缺少库: {e}"); sys.exit(1)

dashscope.api_key = API_KEY

# ================= 功能函数 =================

def get_embedding(texts):
    if not texts: raise ValueError("文本为空")
    clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts: raise ValueError("所有文本均为空")
    
    batch_size = 20
    all_embeddings = []
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i+batch_size]
        resp = dashscope.TextEmbedding.call(model=MODEL_EMBED, input=batch)
        if resp.status_code != 200:
            raise Exception(f"API 失败: {resp.message}")
        embeddings = [item['embedding'] for item in resp.output['embeddings']]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings, dtype='float32')

def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i+1, "text": text})
    return pages

def split_text(pages, chunk_size=500):
    chunks = []
    for page in pages:
        text = page['text']
        if len(text) <= chunk_size:
            chunks.append({"content": text.strip(), "metadata": {"page": page['page']}})
        else:
            paragraphs = text.split('\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk.strip():
                        chunks.append({"content": current_chunk.strip(), "metadata": {"page": page['page']}})
                    current_chunk = para
                else:
                    current_chunk += "\n" + para
            if current_chunk.strip():
                chunks.append({"content": current_chunk.strip(), "metadata": {"page": page['page']}})
    return chunks

def save_index(chunks, embeddings, base_path):
    """保存到指定路径，自动添加后缀"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 生成完整文件名
    faiss_file = base_path + ".faiss"
    data_file = base_path + ".data"
    
    faiss.write_index(index, faiss_file)
    with open(data_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"   💾 已保存: {os.path.basename(faiss_file)} & {os.path.basename(data_file)}")
    return index

def load_index(base_path):
    """从指定路径加载"""
    faiss_file = base_path + ".faiss"
    data_file = base_path + ".data"
    
    if not os.path.exists(faiss_file) or not os.path.exists(data_file):
        return None, None
        
    index = faiss.read_index(faiss_file)
    with open(data_file, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def search(index, chunks, query_vector, k=3):
    if index is None:
        raise RuntimeError("向量索引未加载")
    D, I = index.search(query_vector.reshape(1, -1), k)
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

def ask_llm(context, question):
    prompt = f"""根据以下资料回答问题：
【资料】: {context}
【问题】: {question}
【回答】: """
    resp = dashscope.Generation.call(model=MODEL_LLM, prompt=prompt, api_key=API_KEY)
    if resp.status_code == 200:
        return resp.output.text
    return f"❌ LLM 错误: {resp.message}"

# ================= 主流程 =================

# 1. 尝试加载 (使用新的路径)
index, chunks = load_index(INDEX_PATH)

if index is None:
    print(f"📄 未找到索引，正在处理 {PDF_FILE} ...")
    if not os.path.exists(PDF_FILE):
        print(f"❌ 文件不存在: {PDF_FILE}"); sys.exit(1)
    
    pages = read_pdf(PDF_FILE)
    if not pages:
        print("❌ PDF 内容为空"); sys.exit(1)
        
    chunks = split_text(pages)
    if not chunks:
        print("❌ 切分后无内容"); sys.exit(1)
    
    print(f"   切分为 {len(chunks)} 个片段，正在向量化...")
    texts = [c['content'] for c in chunks]
    embeddings = get_embedding(texts)
    
    print("   💾 正在保存索引到 faiss_index 目录...")
    index = save_index(chunks, embeddings, INDEX_PATH)
    print(f"✅ 索引已建立并保存 (内存中已就绪)")
else:
    print(f"✅ 已从 faiss_index 目录加载现有索引 ({len(chunks)} 个片段)")

if index is None:
    print("❌ 严重错误：索引仍未加载"); sys.exit(1)

print("\n🤖 系统就绪！(输入 quit 退出)")

while True:
    try:
        query = input("\n你: ").strip()
        if not query: continue
        if query.lower() in ['quit', 'exit']: break
        
        q_vec = get_embedding([query])[0]
        docs = search(index, chunks, q_vec, k=3)
        context = "\n\n---\n\n".join([f"[页码:{d['metadata'].get('page','?')}] {d['content']}" for d in docs])
        
        print("🤖 思考中...", end='\r')
        answer = ask_llm(context, query)
        print(" " * 20, end='\r')
        print(f"🤖 AI: {answer}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ 出错: {e}")