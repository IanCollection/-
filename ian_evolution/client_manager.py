from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
minimax_client = OpenAI(api_key=os.environ.get("MINIMAX_API_KEY"), base_url="https://api.minimax.chat/v1")
# deepseek_client = OpenAI(api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.siliconflow.cn/v1")

qwen_client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

siliconflow_client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"), base_url="https://api.siliconflow.cn/v1")
