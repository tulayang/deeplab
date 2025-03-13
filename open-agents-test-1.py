import asyncio
import os

from openai import AsyncOpenAI
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

client = AsyncOpenAI(api_key="<API key>", base_url="http://<Host>:<Port>/v1")

class CustomModelProvider(ModelProvider):
  def get_model(self, model_name: str | None) -> Model:
    return OpenAIChatCompletionsModel(model='DeepSeek-R1', openai_client=client)
  
CUSTOM_MODEL_PROVIDER = CustomModelProvider()

@function_tool
def get_weather(city: str):
  print(f"[debug] getting weather for {city}")
  return f"The weather in {city} is sunny."

async def main():
  agent = Agent(name="Assistant", instructions="You are a helpful assistant.", tools=[get_weather])
  result = await Runner.run(
      agent,
      "Write a haiku about recursion in programming.",
      run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
  )
  print(result.final_output)

asyncio.run(main())