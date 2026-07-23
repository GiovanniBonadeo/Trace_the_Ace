"""
check_thinking.py

Sends a small, simple question to your server with a generous token budget
and prints the FULL raw response, so you can see directly whether the model
is generating hidden reasoning even though enable_thinking is set to False.

Run:  python check_thinking.py
"""

import os
import json
import requests

LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://10.70.13.33:11434")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "sk-RZSBTkuZYOeXULKBTKupkA")
MODEL_NAME = "qwen/qwen3.5-35b-a3b"


def call(enable_thinking: bool, max_tokens: int = 500) -> dict:
    base = LOCAL_API_URL.rstrip("/")
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Follow the requested JSON output exactly."},
            {"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"status": "ok"}'},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {"Content-Type": "application/json"}
    if LOCAL_API_KEY and LOCAL_API_KEY != "your-local-api-key":
        headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def report(label: str, data: dict) -> None:
    print(f"\n===== {label} =====")
    choice = data["choices"][0]
    message = choice.get("message", {})
    print("finish_reason:", choice.get("finish_reason"))
    print("usage:", data.get("usage"))
    print("content:", repr(message.get("content")))
    for key in ("reasoning_content", "reasoning", "thinking"):
        if key in message:
            print(f"{key} (first 300 chars):", repr(str(message[key])[:300]))
    print("\nFULL RAW RESPONSE:")
    print(json.dumps(data, indent=2)[:3000])


def main():
    print(f"Testing model={MODEL_NAME} at {LOCAL_API_URL}")
    print("This uses a trivial prompt with max_tokens=500, so if the model is")
    print("reasoning heavily, it should hit the token limit even on an easy question.\n")

    data_off = call(enable_thinking=False)
    report("enable_thinking = False", data_off)

    data_on = call(enable_thinking=True)
    report("enable_thinking = True (for comparison)", data_on)

    print("\n===== VERDICT =====")
    content_off = data_off["choices"][0]["message"].get("content") or ""
    finish_off = data_off["choices"][0].get("finish_reason")
    if content_off.strip() and finish_off != "length":
        print("Looks OK: got real content back with thinking disabled.")
    else:
        print("PROBLEM CONFIRMED: even with enable_thinking=False and a trivial")
        print("prompt, content is empty or got cut off. Your server/model is not")
        print("actually honoring the 'disable thinking' setting sent this way.")


if __name__ == "__main__":
    main()
