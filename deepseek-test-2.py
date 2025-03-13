from openai import OpenAI

# 1. 对话 - 非流式
################################################## 

client = OpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")

# 第一回合
messages=[
  {"role": "user", "content": "8.16 和 8.9，哪个更大？"}
]
response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=messages,
  stream=False
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(response)
print(reasoning_content)
print(content)
print('=================================================')

# 第二回合 - 在多回合对话中拼接上下文
messages.append({'role': 'assistant', 'content': content})
messages.append({'role': 'user', 'content': "'走南闯北' 这个词有几个表示方向的字？"})
response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=messages,
  stream=False
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(response)
print(reasoning_content)
print(content)
print('=================================================')
