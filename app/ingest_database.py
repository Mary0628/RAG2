from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import os



from dotenv import load_dotenv

load_dotenv()


# 修改为临时路径（Railway兼容）
import tempfile
import os

# 使用系统临时目录
DATA_PATH = os.path.join(tempfile.gettempdir(), "data")
CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")

# 确保目录存在
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)




DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

import logging

logging.basicConfig(level=logging.INFO)


def ingest_documents():
    try:
        logging.info("开始文档分析流程")

        # 加载模型部分添加日志
        logging.info("正在加载嵌入模型...")
        embeddings_model = HuggingFaceEmbeddings(
            #model_name="./models/all-MiniLM-L6-v2"
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

        # 数据库操作日志
        if os.path.exists(CHROMA_PATH):
            logging.info("检测到现有数据库，准备清空集合...")
            existing_store = Chroma(
                collection_name="example_collection",
                embedding_function=embeddings_model,
                persist_directory=CHROMA_PATH,
            )
            existing_store.delete_collection()
            logging.info("旧集合已清空")
        else:
            logging.info("未找到现有数据库，将新建数据库")

        # 文档处理日志
        logging.info(f"正在从 {DATA_PATH} 加载文档...")
        loader = PyPDFDirectoryLoader(DATA_PATH)
        raw_documents = loader.load()
        logging.info(f"成功加载 {len(raw_documents)} 个原始文档")

        # 添加异常捕获
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(raw_documents)
            logging.info(f"文档分割完成，生成 {len(chunks)} 个文本块")
        except Exception as e:
            logging.error(f"文档分割失败: {str(e)}")
            raise

        # 数据库写入日志
        logging.info("正在初始化向量数据库...")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings_model,
            persist_directory=CHROMA_PATH,
        )

        logging.info("正在生成UUID...")
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        try:
            logging.info("正在写入数据库...")
            vector_store.add_documents(documents=chunks, ids=uuids)
            logging.info("数据库写入成功")
        except Exception as e:
            logging.error(f"数据库写入失败: {str(e)}")
            raise

    except Exception as e:
        logging.error(f"流程执行失败: {str(e)}")
        raise