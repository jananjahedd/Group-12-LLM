from openai import OpenAI
import os
from pathlib import Path
import pandas as pd


root = Path(__file__).resolve().parent.parent.parent


MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

data_path = root / 'data' / 'train-balanced-sarcasm.csv'

data = pd.read_csv(data_path)
comments = data['comment'].head(10)

comments_text = "\n".join(comments.fillna(''))

input_text = f"Here are 3 comments:\n{comments_text}\nPlease identify any sarcastic remarks."

completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": input_text}
    ]
)

print("Assistant: " + completion.choices[0].message.content)
