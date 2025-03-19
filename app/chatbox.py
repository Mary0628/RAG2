import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import os
import shutil
from openai import OpenAI
import socket
import random
from datetime import datetime
import csv
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
# 修改为临时路径（Railway兼容）
import tempfile
import os

# 使用系统临时目录
DATA_PATH = os.path.join(tempfile.gettempdir(), "data")
CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")

# 确保目录存在
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
PASSWORD = "123456"

# 配置新增
FEEDBACK_PATH = r"feedback_data"
CSV_FILE = Path(FEEDBACK_PATH) / "feedback.csv"

# 设置本地存储
def setup_local_storage():
    try:
        # 创建反馈目录
        Path(FEEDBACK_PATH).mkdir(parents=True, exist_ok=True)

        # 初始化CSV文件（如果不存在）
        if not CSV_FILE.exists():
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["UID", "Timestamp", "Feedback Type", "Input Question", "LLM Answer"])

        print("Local storage setup complete")
        return CSV_FILE
    except Exception as e:
        print(f"Error setting up local storage: {e}")
        return None


try:
    csv_file = setup_local_storage()
    print("Setting up local storage")
except Exception as e:
    print(f"Error setting up local storage: {e}")
    csv_file = None


# 修改后的反馈处理函数
# 修改文档1中的handle_feedback函数（保留第一个完整定义，删除第二个不完整的定义）
def handle_feedback(feedback_type, history):
    try:
        uid = str(int(time.time()))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not history or len(history) == 0:
            return "No history available for feedback"

        # 确保正确处理历史记录格式
        last_interaction = history[-1]
        if isinstance(last_interaction, list) and len(last_interaction) >= 2:
            last_question = last_interaction[0]
            last_answer = last_interaction[1]
        else:
            return "Invalid history format"

        # 确保目录存在
        Path(FEEDBACK_PATH).mkdir(parents=True, exist_ok=True)

        # 写入CSV文件（使用追加模式）
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([uid, timestamp, feedback_type, last_question, last_answer])

        return f"Feedback recorded: {feedback_type}"

    except Exception as e:
        return f"Error saving feedback: {str(e)}"



embeddings_model = HuggingFaceEmbeddings(
    #model_name="./models/all-MiniLM-L6-v2"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


# Initialize OpenAI client (pointing to DeepSeek)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Connect to ChromaDB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})


# Function to find an available port
# def find_available_port(start=5001, end=5010):
#     while start <= end:
#         try:
#             sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             sock.bind(('0.0.0.0', start))
#             sock.close()
#             return start
#         except OSError:
#             start += 1
#     raise OSError("No available ports in range")
# 替换 find_available_port 函数为：
def find_available_port():
    return int(os.environ.get("PORT", 5001))

# Upload documents function
def upload_files(files):
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)
    for file in files:
        shutil.copy(file.name, DATA_PATH)
    return "Documents uploaded successfully!"


# Password verification function
def check_password(password):
    if password == PASSWORD:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            "Password correct!"
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "Incorrect password!"
        )


# Document analysis function
def analyze_documents():
    try:
        from ingest_database import ingest_documents
        import traceback

        # Clear previous database connection
        global vector_store
        if 'vector_store' in globals():
            del vector_store

        # Perform analysis
        ingest_documents()

        # Reinitialize database connection
        global retriever
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings_model,
            persist_directory=CHROMA_PATH,
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

        return "Document analysis completed!"

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def handle_feedback(feedback_type, history):
    try:
        uid = str(int(time.time()))

        print(f"Got feedback: {feedback_type}")

        if not history or len(history) == 0:
            print("No history, no feedback")
            return "No history"

        last_interaction = history[-1]
        last_question = last_interaction[0] if last_interaction else ""
        last_answer = last_interaction[1] if len(last_interaction) > 1 else ""

        print(f"Last Question: {last_question}")
        print(f"Last Answer: {last_answer}")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        row_data = [uid, timestamp, feedback_type, last_question, last_answer]

        try:
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([uid, timestamp, feedback_type, last_question, last_answer])
        except Exception as e:
            print(f"写入CSV时出错: {e}")

        # print(f"Successfully stored feedback - UID: {uid}")
        # return f"Feedback recorded (ID: {uid})"

    except Exception as e:
        error_msg = f"Failed to store feedback: {str(e)}"
        print(error_msg)
        return error_msg


# Question answering function
def stream_response(message, history):
    try:
        docs = retriever.invoke(message)
        knowledge = "\n\n".join([doc.page_content for doc in docs])
        rag_prompt = f"""
        You are an assistant, which answers questions based on knowledge which is provided to you. While answering, you don't use your internal knowledge, but solely the information in the "The knowledge" section. You don't mention anything to the user about the provided knowledge.
        You first greet the teachers, Welcome to the Learning Design Studio!  then start the conversation. 
        You wait to guide teachers to design good well-written learning outcomes and evaluate whether the learning outcomes teachers design meet the standards of good well-written learning outcomes and give some good examples if needed.

            (1)Please make sure the design of learning outcomes should be learner centered.

            (2)Please make sure the learning outcomes should begin with a verb such as create a prototype of a sustainable water filtration system

            (3)If the teachers want to assess whether the learning outcomes is good enough, please help teachers assess whether the learning outcomes teachers provided meet the standards of good well-written learning outcomes and classify the learning outcomes into good learning outcomes and bad learning outcomes based on the standards and examples of different learning outcomes. 

            (4) If the learning outcomes is good enough, please just give the positive feedback such as "Good job" without further information. 

            (5) If the learning outcomes do not meet the standards, please give some good examples of good well-written learning outcomes.
        Please remember to answer the questions in the following structured format:
        1. Start with Summary (without using ** marks)
        2. Follow with Key Points (bullet points)
        3. End with Examples (if applicable)

        Question: {message}
        History: {history}
        Knowledge: {knowledge}
        """
        partial_message = ""
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": rag_prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

#         # 修改后的反馈按钮HTML
#         feedback_buttons = """
#
# # ---
# # Was this response:
# # <div style='display: flex; gap: 10px; margin-top: 15px; margin-bottom: 10px;'>
# #     <button style='padding: 5px 15px; border: 1px solid #4A90E2; border-radius: 5px; color: #4A90E2; background: white; cursor: pointer; margin-right: 10px;' onclick='document.querySelector("#helpful-btn").click()'>Helpful</button>
# #     <button style='padding: 5px 15px; border: 1px solid #4A90E2; border-radius: 5px; color: #4A90E2; background: white; cursor: pointer; margin-right: 10px;' onclick='document.querySelector("#not-helpful-btn").click()'>Not Helpful</button>
# #     <button style='padding: 5px 15px; border: 1px solid #4A90E2; border-radius: 5px; color: #4A90E2; background: white; cursor: pointer;' onclick='document.querySelector("#incorrect-btn").click()'>Incorrect</button>
# # </div>
# # """
#         yield partial_message + feedback_buttons

    except Exception as e:
        yield f"System error: {str(e)}"


def respond(message, chat_history):
    response = ""
    for chunk in stream_response(message, chat_history):
        response = chunk

    chat_history.append((message, response))
    return "", chat_history


# Gradio interface
with gr.Blocks(theme="soft") as demo:

    gr.Markdown("""
        <div style="margin-bottom: 1rem; font-family: 'Times New Roman', Times, serif">
            <h1 style="color: #2563eb; margin-bottom: 0.5rem; text-align: center">AI Chatbot</h1>
            <p style="color: #4b5563; font-size: 1.1rem; text-align: left">
                This is an AI-based Chatbot that can：<br>
                • Answer your questions related to learning outcomes<br>
                • Help you design good well-written learning outcomes<br>
                • Help you assess whether the learning outcomes you provide meet the standards of good well-written learning outcomes<br>
                Please enter your question to start the conversation!
            </p>
        </div>
    """)

    # Main chat interface
    with gr.Column() as chat_column:
        chatbot = gr.Chatbot(
            [],
            bubble_full_width=False,
            avatar_images=(None, None),  # 移除头像
            render_markdown=True,
            height=400,
            show_label=False,  # 隐藏标签
            show_copy_button=False,  # 隐藏复制按钮
            layout="bubble"  # 使用气泡布局
        )
        # with gr.Column(visible=False):
        #     helpful_btn = gr.Button("Helpful", variant="secondary", size="sm", elem_id="helpful-btn")
        #     not_helpful_btn = gr.Button("Not Helpful", variant="secondary", size="sm", elem_id="not-helpful-btn")
        #     incorrect_btn = gr.Button("Incorrect", variant="secondary", size="sm", elem_id="incorrect-btn")
            # 创建按钮时绑定处理函数
        gr.Markdown("""
                <div style="margin-bottom: 1rem; font-family: 'Times New Roman', Times, serif">
                    <p style="color: #4b5563; font-size: 1.1rem; text-align: left">
                        Was this response:<br>
                    </p>
                </div>
            """)
        with gr.Row():  # 紧密排列
            helpful_btn = gr.Button(
                "Helpful",
                size="sm",
                variant="secondary",
                min_width=50,
                elem_id="custom-helpful-btn"
            )
            not_helpful_btn = gr.Button(
                "Not Helpful",
                size="sm",
                variant="secondary",
                min_width=50
            )
            incorrect_btn = gr.Button(
                "Incorrect",
                size="sm",
                variant="stop",
                min_width=50
            )



        with gr.Row():
            msg = gr.Textbox(
                label="Question Input",
                placeholder="Type a message...",
                container=False,
                scale=7
            )
            # feedback buttons
            submit_btn = gr.Button("Send", variant="primary")
            # 反馈按钮行
        # -------------------------------
        # feedback_status = gr.Textbox(label="Feedback Status", visible=False, interactive=False)
        # 增加反馈结果显示组件
        feedback_output = gr.Textbox(label="Feedback Status", visible=False)


        # 绑定点击事件
        helpful_btn.click(
            fn=handle_feedback,
            inputs=[gr.State("helpful"), chatbot],
            outputs=feedback_output
        )
        not_helpful_btn.click(
            fn=handle_feedback,
            inputs=[gr.State("not_helpful"), chatbot],
            outputs=feedback_output
        )
        incorrect_btn.click(
            fn=handle_feedback,
            inputs=[gr.State("incorrect"), chatbot],
            outputs=feedback_output
        )
        gr.Examples(
            examples=[
                "What is learning outcome",
                "What are well-written learning outcomes",
                "Can you give me examples of learning outcomes",
            ],
            inputs=msg,
            label="Examples",
            examples_per_page=4
        )

        with gr.Row():
            Navigation_btn = gr.Button(
                "Navigation Chatbot",
                size="lg",
                variant="secondary",
                min_width=100,
                elem_id="custom-Navigation-btn"
            )
            Assessment_btn = gr.Button(
                "Assessment Chatbot",
                size="lg",
                variant="secondary",
                min_width=100,
                elem_id="custom-Assessment-btn"
            )
        # 正确的事件绑定位置
        Navigation_btn.click(
            fn=None,
            js="""() => {
                console.log('跳转触发');  // 调试日志
                window.top.location.href = 'https://chatbot-test2-six.vercel.app/';  // 处理iframe嵌套
            }"""
        )
        Assessment_btn.click(
            fn=None,
            js="""() => {
                        console.log('跳转触发');  // 调试日志
                        window.top.location.href = 'http://39.104.16.59:5001/';  // 处理iframe嵌套
                    }"""
        )

    # Hidden components
    with gr.Column(visible=False) as hidden_components:
        password_input = gr.Textbox(
            label="Password Input",
            type="password",
            placeholder="Enter password..."
        )
        password_button = gr.Button("Verify Password")
        password_output = gr.Textbox(
            label="Verification Result",
            interactive=False
        )
        operation_guide = gr.Markdown()

        with gr.Column() as upload_row:
            file_upload = gr.Files(
                label="Upload Documents",
                file_types=[".pdf"]
            )
            upload_button = gr.Button(
                "Upload Documents",
                variant="primary"
            )
        upload_output = gr.Textbox(
            label="Upload Result",
            interactive=False
        )

        with gr.Column() as analyze_row:
            analyze_button = gr.Button(
                "Analyze Documents",
                variant="primary"
            )
        analyze_output = gr.Textbox(
            label="Analysis Result",
            interactive=False
        )

    # Event bindings
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])

    # # binding feedback buttons
    # helpful_btn.click(
    #     handle_feedback,
    #     inputs=[gr.Textbox(value="helpful", visible=False), chatbot],
    #     outputs=feedback_status
    # )
    #
    # not_helpful_btn.click(
    #     handle_feedback,
    #     inputs=[gr.Textbox(value="not helpful", visible=False), chatbot],
    #     outputs=feedback_status
    # )
    #
    # incorrect_btn.click(
    #     handle_feedback,
    #     inputs=[gr.Textbox(value="incorrect", visible=False), chatbot],
    #     outputs=feedback_status
    # )

    password_button.click(
        check_password,
        inputs=password_input,
        outputs=[
            operation_guide,
            upload_row,
            analyze_row,
            upload_output,
            analyze_output,
            password_output
        ]
    )
    upload_button.click(
        upload_files,
        inputs=file_upload,
        outputs=upload_output
    )
    analyze_button.click(
        analyze_documents,
        outputs=analyze_output
    )

# Launch application
# if __name__ == "__main__":
#     try:
#         available_port = find_available_port()
#         demo.launch(
#             server_name="0.0.0.0",
#             server_port=available_port,
#             share=True,
#             debug=True,
#             quiet=False
#         )
#         print(f"Application launched on port {available_port}")
#     except Exception as e:
#         print(f"Error launching application: {e}")
if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("PORT", 5001)))