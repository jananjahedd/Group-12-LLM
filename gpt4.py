from openai import OpenAI
import os


MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "user", "content": "Hello! Could you solve 2+2?"} 
  ]
)

print("Assistant: " + completion.choices[0].message.content)
