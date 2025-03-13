import re
from openai import OpenAI

# 1. 对话 - 流式
################################################## 

client = OpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")

# 第一回合
messages=[
  {"role": "user", "content": "散文有什么特点？"}
]
response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=messages,
  stream=True
)

reasoning_content = ""
content = ""

# 当 stream=True 时，openai 似乎不能正确解析 reasoning_content
# 所以，我们只好手动解析 - 通过一个迷你状态机

STATE_INIT = 1
STATE_REASON = 2
STATE_CONTENT = 3

# 解析对话
state = STATE_INIT
for chunk in response:
  if hasattr(chunk.choices[0].delta, "content"):
    token = chunk.choices[0].delta.content
    if state == STATE_INIT:
      if token == '<think>':
        state = STATE_REASON
      elif not re.match(r"^([\s\r\n]*)$", token):
        state = STATE_CONTENT
        header_content += token
    elif state == STATE_REASON:
      if token == '</think>':
        state = STATE_CONTENT
      else:
        reasoning_content += token
        print(f'reasoning_content: {reasoning_content}')
    elif state == STATE_CONTENT:
      content += token
      print(f'content: {content}')

print('=================================================')

# 第二回合 - 在多回合对话中拼接上下文
messages.append({'role': 'assistant', 'content': content})
messages.append({'role': 'user', 'content': "'我怎么写散文？"})
response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=messages,
  stream=True
)

reasoning_content = ""
content = ""
reasoning_flag = False

# 解析对话
state = STATE_INIT
for chunk in response:
  if hasattr(chunk.choices[0].delta, "content"):
    token = chunk.choices[0].delta.content
    if state == STATE_INIT:
      if token == '<think>':
        state = STATE_REASON
      elif not re.match(r"^([\s\r\n]*)$", token):
        state = STATE_CONTENT
        header_content += token
    elif state == STATE_REASON:
      if token == '</think>':
        state = STATE_CONTENT
      else:
        reasoning_content += token
        print(f'reasoning_content: {reasoning_content}')
    elif state == STATE_CONTENT:
      content += token
      print(f'content: {content}')

print('=================================================')