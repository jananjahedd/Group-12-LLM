"""
Author: Janan Jahed (s5107318), Andrei Medesan (), Alex Cernat()
File: gemini.py
Description: Accessing the gemini-1.5-flash model through the ai studio
using personalised API to fine tune later.
"""
import os
import google.generativeai as genai
from pathlib import Path
import pandas as pd


root = Path(__file__).resolve().parent.parent.parent

# https://aistudio.google.com/u/1/prompts/1M9GUN91npG1SrTFa5CFIYcrEco0WZJdd
# API key added to the shell - AIzaSyD57QVLY57pt6UphqMnm5iGz2i_Uff0Tik
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

data_path = root / 'data' / 'train-balanced-sarcasm.csv'
data = pd.read_csv(data_path)
comments = data['comment'].head(3)

comments_text = "\n".join(comments.fillna(''))

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

input_text = f"Here are 3 comments:\n{comments_text}\nPlease identify any sarcastic remarks."

response = model.generate_content([
    input_text,
])

# Inspect the response again
if response.candidates:
    for i, candidate in enumerate(response.candidates):
        print(f"Candidate {i}: {candidate}")
else:
    print("No valid response generated.")

print(response.text)