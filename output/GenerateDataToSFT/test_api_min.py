# -*- coding: utf-8 -*-
import os
import time
from openai import OpenAI


API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-HUnMwzbL0kiac1TnUkuk6BunochazpWF32mgTHM5nGDqeaoo")
BASE_URL = "https://yinli.one/v1"
MODEL_NAME = "deepseek-r1"

if not API_KEY:
    raise ValueError("Please set DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60.0,
    max_retries=0,
)

start = time.time()

print("calling api...", flush=True)

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "Return only: ok"}
    ],
    temperature=0,
    max_tokens=10,
    timeout=60.0,
)

elapsed = time.time() - start

print("api returned")
print("elapsed_seconds:", round(elapsed, 2))
print("content:", resp.choices[0].message.content)
