"""
test_server.py — quick smoke test for your local server + model.

Run:  python test_server.py
"""

import os
import requests

LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://10.70.13.33:11434")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "sk-RZSBTkuZYOeXULKBTKupkA")
MODEL_NAME = "qwen/qwen3.5-35b-a3b"


def main() -> None:
    base = LOCAL_API_URL.rstrip("/")
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Follow the requested JSON output exactly."},
            {"role": "user", "content": "Reply with exactly: {\"status\": \"ok\"}"},
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if LOCAL_API_KEY and LOCAL_API_KEY != "your-local-api-key":
        headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"

    print(f"Calling {url} with model={MODEL_NAME} ...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"CONNECTION FAILED: {e}")
        return

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print("Server returned an error. Raw response:")
        print(response.text)
        return

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if content:
        print("SUCCESS. Model response:")
        print(content.strip())
    else:
        print("Response was 200 but content was empty. Raw JSON:")
        print(data)


if __name__ == "__main__":
    main()
