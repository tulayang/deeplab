# 上下文长度：64K     
# 最大思维链长度：32K 
# 最大输出长度：8K    
#
# Temperature：默认 1.0 -- 建议：
#
#   代码生成/数学解题   - 0.0
#   数据抽取/分析       - 1.0
#   通用对话            - 1.3
#   翻译                - 1.3
#   创意类写作/诗歌创作 - 1.5
#
# 限速：
#
#   - DeepSeek API 不限制用户并发量，我们会尽力保证您所有请求的服务质量。

import os
from openai import OpenAI

# 1. 基本调用
################################################## 
#
# curl http://<Host>:<Port>/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer <API key>" \
#   -d '{
#         "model": "DeepSeek-R1",
#         "messages": [
#           {"role": "system", "content": "你是个懂编程的助理。"},
#           {"role": "user", "content": "你是谁!"}
#         ],
#         "stream": false
#       }'
#
# {
#   "id": "chatcmpl-5e4bfedcdb41453dabe2254cf934cdc6",
#   "object": "chat.completion",
#   "created": 1741597864,
#   "model": "DeepSeek-R1",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "reasoning_content": null,
#         "content": "<think>\n...\n</think>\n\n你好！很高兴见到你，...😊",
#         "tool_calls": []
#       },
#       "logprobs": null,
#       "finish_reason": "stop",
#       "stop_reason": null
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 12,
#     "total_tokens": 47,
#     "completion_tokens": 35,
#     "prompt_tokens_details": null
#   },
#   "prompt_logprobs": null
# }(base) 
#
##################################################

client = OpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")

response = client.chat.completions.create(
  model="DeepSeek-R1",
  messages=[
    {"role": "system", "content": "你是个懂编程的助理。"},
    {"role": "user", "content": "你好!"}
  ],
  stream=False
)

print(response)
print(response.choices[0].message.content)

# 2. 列出模型
################################################## 
# 
# curl -L -X GET 'http://<Host>:<Port>/v1/models' \
# -H 'Accept: application/json' \
# -H 'Authorization: Bearer deepseek-hello'
#
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "DeepSeek-R1",
#       "object": "model",
#       "created": 1741598218,
#       "owned_by": "vllm",
#       "root": "/sdses02/zl/deepseek/DeepSeek-R1",
#       "parent": null,
#       "max_model_len": 16384,
#       "permission": [
#         {
#           "id": "modelperm-d72d2737da134743a2cae4988abc2b4e",
#           "object": "model_permission",
#           "created": 1741598218,
#           "allow_create_engine": false,
#           "allow_sampling": true,
#           "allow_logprobs": true,
#           "allow_search_indices": false,
#           "allow_view": true,
#           "allow_fine_tuning": false,
#           "organization": "*",
#           "group": null,
#           "is_blocking": false
#         }
#       ]
#     }
#   ]
# }(base)
#
##################################################

client = OpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")
print(client.models.list())