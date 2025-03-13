from openai import OpenAI

# 1. 函数调用
#
#    当前版本 deepseek-chat 模型 Function Calling 功能效果不稳定，会出现
#    循环调用、空回复的情况。我们正在积极修复中，预计将在下一个版本中得到修复。
################################################## 

client = OpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather of an location, the user shoud supply a location first",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          }
        },
        "required": ["location"]
      },
    }
  }
]

messages=[
  {"role": "user", "content": "How's the weather in Hangzhou?"}
]

response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=messages,
  tools=tools,
  tool_choice={"type": "function", "function": {"name": "get_weather"}},
  stream=False
)