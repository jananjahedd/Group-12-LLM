"""
Author: Janan Jahed (s5107318), Andrei Medesan (), Alex Cernat()
File: gemini.py
Description: Accessing the gemini-1.5-flash model through the ai studio
using personalised API to fine tune later.
"""

import os
import google.generativeai as genai

# https://aistudio.google.com/u/1/prompts/1M9GUN91npG1SrTFa5CFIYcrEco0WZJdd
# API key added to the shell - AIzaSyD57QVLY57pt6UphqMnm5iGz2i_Uff0Tik
genai.configure(api_key=os.environ["API_KEY"])

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

response = model.generate_content([
  "input: Write a short story about a robot in the future.",
  "output: ",
])

print(response.text)
