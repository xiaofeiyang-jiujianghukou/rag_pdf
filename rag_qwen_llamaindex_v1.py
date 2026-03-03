import os
import sys
import logging
from pathlib import Path

# ================= LlamaIndex 核心导入 =================
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer

# 第三方库
import faiss

# ================= 配置区域 =================
API_KEY = "sk-a41ebd38c93c43269e46a6ec46787f18"
PDF_FILE = "demo.pdf"
PERSIST_DIR = "./faiss_llamaindex"  # LlamaIndex 通常保存整个存储上下文

# 模型配置
MODEL_EMBED = "text-embedding-v3"
MODEL_LLM = "qwen-plus"

if API_KEY == "你的_阿里云_DashScope_API_KEY":
    if os.getenv("DASHSCOPE_API_KEY"):
        API_KEY = os.getenv("DASHSCOPE_API_KEY")
    else:
        print("❌ 请填入 API_KEY"); sys.exit(1)

# 设置 API Key 环境变量 (LlamaIndex 的 dashscope 包会自动读取)
os.environ["DASHSCOPE_API_KEY"] = API_KEY

# 配置日志 (LlamaIndex 默认日志较多，设为 WARNING)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ================= 初始化全局设置 =================
print("⚙️ 初始化 LlamaIndex + DashScope ...")

# 1. 设置嵌入模型 (使用官方插件)
Settings.embed_model = DashScopeEmbedding(model_name=MODEL_EMBED)

# 2. 设置大语言模型 (使用官方插件)
Settings.llm = DashScope(model=MODEL_LLM, api_key=API_KEY)

# 3. 设置文本切分策略 (Node Parser)
# LlamaIndex 称之为 NodeParser，默认按句子切分，更智能
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# 4. 设置上下文窗口 (防止截断)
Settings.context_window = 4096

# ================= 主流程 =================

def build_index():
    """构建或加载索引"""
    
    # 检查是否存在已保存的索引
    if not Path(PERSIST_DIR).exists():
        print(f"📄 未找到索引，正在处理 {PDF_FILE} ...")
        
        if not os.path.exists(PDF_FILE):
            print(f"❌ 文件不存在: {PDF_FILE}"); sys.exit(1)

        # 1. 加载文档 (LlamaIndex 自动处理 PDF 解析)
        # 注意：SimpleDirectoryReader 需要文件夹路径，我们把单个文件放入临时逻辑或当前目录
        documents = SimpleDirectoryReader(input_files=[PDF_FILE]).load_data()
        print(f"   ✅ 成功加载 {len(documents)} 个文档对象")

        # 2. 创建向量索引 (自动完成切分、向量化)
        print("   🔢 正在构建向量索引 (这可能需要一点时间)...")
        index = VectorStoreIndex.from_documents(documents)
        
        # 3. 持久化存储
        print(f"   💾 正在保存索引到 {PERSIST_DIR} ...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("   ✅ 索引保存成功")
        
        return index
    else:
        print(f"📂 发现现有索引，正在从 {PERSIST_DIR} 加载 ...")
        # 加载存储上下文
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        # 从存储上下文加载索引
        index = load_index_from_storage(storage_context)
        print("   ✅ 索引加载成功")
        return index

def run_chat(index):
    """运行问答循环 - 增强版"""
    
    # 1. 配置检索器
    # 增加 top_k 到 5，确保即使相关性低也能捞几个片段回来
    retriever = index.as_retriever(similarity_top_k=5)
    
    # 2. 定义自定义提示词模板
    qa_prompt_tmpl = """\
你是一个基于文档的智能助手。请仅根据提供的【上下文信息】回答问题。

【重要规则】：
1. 如果【上下文信息】为空，或者不包含问题的答案，你必须直接回答：“抱歉，文档中没有提到关于这个问题的信息。”
2. 严禁利用你自己的外部知识（如天气、新闻等）来回答。
3. 严禁编造事实。

【上下文信息】：
---------------------
{context_str}
---------------------

问题：{query_str}
回答：
"""
    qa_prompt = PromptTemplate(qa_prompt_tmpl)

    # 3. 创建响应合成器 (Response Synthesizer)
    # 显式创建合成器并注入提示词，这比 update_prompts 更稳健
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
        # 关键参数：即使没有检索到节点，也允许合成器运行（传入空上下文）
        # 在某些版本中，这能防止直接返回空
    )

    # 4. 创建查询引擎
    # ⚠️ 注意：这里我们注释掉了 similarity_cutoff，或者设得非常低 (0.1)
    # 目的是：即使是不相关的内容，也要传给 LLM，让 LLM 去判断并拒绝，而不是由过滤器直接拦截
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)] # 设为极低，或者直接注释掉这行测试
        node_postprocessors=[] # 【推荐】暂时完全关闭相似度过滤，测试效果
    )

    print("\n🦙 LlamaIndex 系统就绪！(输入 quit 退出)")
    #print("💡 提示：试着问'明天天气'，看我是否会根据文档拒绝回答。")

    while True:
        try:
            query = input("\n你: ").strip()
            if not query: continue
            if query.lower() in ['quit', 'exit']: break
            
            print("🤖 思考中...", end='\r')
            
            response = query_engine.query(query)
            
            print(" " * 20, end='\r')
            
            # 双重保险：如果 response 还是空的，手动兜底
            answer = str(response).strip()
            if not answer or answer == "None":
                print("🤖 AI: 抱歉，文档中没有找到相关信息，我无法回答这个问题。")
            else:
                print(f"🤖 AI: {answer}")
            
            # 显示来源（如果有）
            # if hasattr(response, 'source_nodes') and response.source_nodes:
            #     pages = [n.metadata.get('page_label', '?') for n in response.source_nodes]
            #     # 只显示前3个来源，避免刷屏
            #     print(f"   📚 来源页码: {pages[:3]}")
            # else:
            #     print("   📚 来源页码: 无 (未检索到相关内容)")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 出错: {e}")
            # 调试时可打开
            # import traceback; traceback.print_exc()

if __name__ == "__main__":
    try:
        index = build_index()
        run_chat(index)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)