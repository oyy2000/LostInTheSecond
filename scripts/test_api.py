#!/usr/bin/env python3
"""Test OpenAI API connectivity with different BASE_URLs."""
from openai import OpenAI

key = 'sk-sNqUCd2XmGHrzXoZDCTmLa7iHdi9L5paDJhTpyNfwkHW8Hc8'

urls = [
    'https://yinli.one/v1',
    'https://yinli.one/',
    'https://yinli.one/v1/chat/completions',
]

for url in urls:
    print(f'Testing: {url}')
    try:
        client = OpenAI(api_key=key, base_url=url)
        resp = client.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': 'What is 2+3? Answer with just the number.'}],
            temperature=0.0,
            max_completion_tokens=16,
        )
        print(f'SUCCESS: {resp.choices[0].message.content}')
        print(f'Model: {resp.model}')
        print(f'WORKING_URL={url}')
        break
    except Exception as e:
        print(f'FAILED: {type(e).__name__}: {e}')
        print()
