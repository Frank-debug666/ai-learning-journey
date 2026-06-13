import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 从环境变量读取密钥，避免将真实 API Key 写进代码。
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一位耐心的 Python 编程老师。"},
        {"role": "user", "content": "请用 Python 输出 1 到 10。"},
    ],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()
