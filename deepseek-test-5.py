# 模拟函数调用
#
# 参考：https://github.com/deepseek-ai/DeepSeek-V2/issues/2#issuecomment-2109475962   

import json
import re
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from openai import OpenAI

def extract_name_and_json(text):
  # 按行分割并去除空行
  #
  # 输入示例：
  #   get_weather
  #   {"location": "北京"}
  # 按行分割，并去除每行首尾空白，同时过滤掉空行
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  
  if len(lines) % 2 != 0:
    raise ValueError("行数不为偶数，可能缺失函数名称或参数")
  
  pairs = []
  # 每两行为一组
  for i in range(0, len(lines), 2):
    name = lines[i]
    json_str = lines[i+1]
    try:
      params = json.loads(json_str)
    except json.JSONDecodeError:
      raise ValueError(f"第 {i+2} 行内容不是有效的 JSON 格式")
    pairs.append((name, params))
  
  return pairs

###############################################################
# 工具函数 1 - 查询函数 - 返回最近一周的网站访问记录
###############################################################

def list_records(parameters):
  return {
    'type': 'json',
    'data': { 
      'visits': [120, 150, 180, 160, 200, 220, 210],
      'days': ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    }
  }

###############################################################
# 工具函数 2 - 绘图函数 - 生成图片
###############################################################

def draw_chart(parameters):
  # 直接指定字体路径
  font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
  font_prop = fm.FontProperties(fname=font_path)

  # 指定支持中文的字体，例如 SimHei，并确保系统已安装该字体
  plt.rcParams['font.family'] = font_prop.get_name()

  # 避免负号显示为方块
  plt.rcParams['axes.unicode_minus'] = False

  # 设置画布大小
  plt.figure(figsize=(10, 5))

  # 绘制折线图，标记点和线条
  plt.plot(parameters['days'], parameters['visits'], marker='o', linestyle='-', color='b')

  # 添加标题和坐标轴标签
  plt.title('最近一周网站访问活跃趋势', fontproperties=font_prop)
  plt.xlabel('日期', fontproperties=font_prop)
  plt.ylabel('访问次数', fontproperties=font_prop)

  # 显示网格
  plt.grid(True)

  # 使用 io.BytesIO 作为缓冲区
  buffer = io.BytesIO()
  
  # 保存图片到缓冲区 (PNG 格式)
  plt.savefig(buffer, format='png', bbox_inches='tight')
  plt.close()  # 关闭图像，释放内存

  # 读取 buffer 内容，并转换为 Base64
  buffer.seek(0)
  img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
  
  return {
    'type': 'image', 
    'data': {
      'content': img_base64
    }
  }

###############################################################
# 工具函数 3 - 天气函数
###############################################################

def get_weather(parameters):
  return {
    'type': 'json',
    'data': {
      'temperature': '28℃'
    }
  }

FUNCTION_MAP = {
  'get_weather': get_weather,
  'list_records': list_records,
  'draw_chart': draw_chart
}

def call_function(name, parameters):
  if name not in FUNCTION_MAP:
    raise ValueError(f'没有匹配的函数： {name}')
  return FUNCTION_MAP[name](parameters)

# 1. 第一回合对话，识别函数名称和参数
###############################################################
client = OpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")

# 由于当前 DeepSeek R1 函数调用不稳定，因此使用提示词替代。由 DeepSeek R1 通过对话确定
# 要调用的函数名称和参数。
messages=[
  {"role": "system", "content": """
可用工具：

[
  {
    "name": "get_weather",
    "description": "获取天气情况，用户应该先指定一个位置",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "位置参数，应该是个城市，比如 '北京'"
        }
      }
    },
    "required": [
      "location"
    ]
  },
  {
    "name": "list_records",
    "description": "获取 Demo 网站访问记录",
    "parameters": {
      "type": "object",
      "properties": {
      }
    }
  },
  {
    "name": "draw_chart",
    "description": "绘制折线图",
    "parameters": {
      "type": "object",
      "properties": {
      }
    }
  }
]

首先查看可用工具，发现有工具可用时，按照以下步骤返回内容：

1. 插入一行文本，内容为 '@tool'，这是用来标记的，不要管它
2. 返回所有工具名称、对应参数
3. 每个工具名称占一行，每个对应参数占一行。对应参数以单行 JSON schema 表示
4. 如果对应参数是空的，返回空 {}
5. 如无提示，每个工具只处理一次
6. 只依照顺序列出即可
7. 无需其他处理

如果没有工具匹配，直接响应。
"""}
]
messages.append({"role": "user", "content": "获取 Demo 网站访问记录，以折线绘制"})

response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=messages,
  stream=True
)

reasoning_content = ""
header_content = ""
content = ""

STATE_INIT = 1
STATE_REASON = 2
STATE_CONTENT = 3
STATE_TEXT = 4
STATE_FUNCTION = 5

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
      header_content += token
      if len(header_content) > 9:
        state = STATE_TEXT
        content += header_content
      elif header_content.strip() == '@tool':
        state = STATE_FUNCTION
    elif state == STATE_TEXT:
      content += token
      print(f'content: {content}')
    elif state == STATE_FUNCTION:
      content += token
      print(f'function: {content}')

# 解析函数调用
if state == STATE_FUNCTION:
  result = []
  pairs = extract_name_and_json(content)
  if len(pairs) > 1:
    first_call = pairs[0]
    call_result = call_function(first_call[0], first_call[1])
    result.append(call_result)
    i = 1
    while i < len(pairs):
      call_result = call_function(pairs[i][0], call_result['data'])
      result.append(call_result)
      i += 1
  print(f'result: {result}')

print('=================================================')

# 第二回合 - 在多回合对话中拼接上下文
# messages.append({'role': 'assistant', 'content': content})
# messages.append({'role': 'user', 'content': "..."})
#
# TODO