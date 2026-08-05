import requests 
import time
import json

from prob_functions import *

def call_server(user_prompt: str, seed: int, local_url, local_key, model_name, temperature, max_tokens, enable_thinking, request_timeout) -> dict:
    base = local_url.rstrip("/")
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
 
    messages = [
        {"role": "system", "content": "Follow the requested JSON output exactly."},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": {
            "think": enable_thinking
        },
    }
    headers = {"Content-Type": "application/json"}
    if local_key and local_key != "your-local-api-key":
        headers["Authorization"] = f"Bearer {local_key}"
 
    response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def call_with_retries(rendered_prompt: str, base_seed: int, retries, retry_backoff_seconds, local_url, local_key, model_name, temperature, max_tokens, enable_thinking, request_timeout) -> dict:
    """Returns a result dict with keys: probability, raw_content, error, elapsed_sec, attempts."""
    last_error = None
    last_content = None
 
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            server_response = call_server(rendered_prompt, seed=base_seed + attempt, local_url=local_url, local_key=local_key, model_name=model_name, temperature=temperature, max_tokens=max_tokens, enable_thinking=enable_thinking, request_timeout=request_timeout)
            content = extract_content(server_response)
            last_content = content
            if not content:
                diagnosis = diagnose_empty_response(server_response)
                print(f"    empty content on attempt {attempt + 1}. {diagnosis}")
            probability = validate_and_extract_probability(content)
            return {
                "probability": probability,
                "raw_content": content,
                "error": None,
                "elapsed_sec": round(time.time() - t0, 2),
                "attempts": attempt + 1,
            }
        except requests.exceptions.RequestException as e:
            last_error = f"RequestException: {e}"
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
 
        if attempt < retries:
            sleep_time = min(retry_backoff_seconds * (2 ** attempt), 10)
            print(f"    attempt {attempt + 1} failed ({last_error}); retrying in {sleep_time}s...")
            time.sleep(sleep_time)
 
    return {
        "probability": None,
        "raw_content": last_content,
        "error": last_error,
        "elapsed_sec": round(time.time() - t0, 2),
        "attempts": retries + 1,
    }
 
 
 
def extract_content(server_response: dict) -> str:
    try:
        return server_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""
 
 
def diagnose_empty_response(server_response: dict) -> str:
    """Build a short diagnostic string explaining why content came back empty."""
    try:
        choice = server_response["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "unknown")
        has_reasoning_field = any(
            k in message for k in ("reasoning_content", "reasoning", "thinking")
        )
        reasoning_preview = ""
        for k in ("reasoning_content", "reasoning", "thinking"):
            if message.get(k):
                reasoning_preview = str(message[k])[:200]
                break
        usage = server_response.get("usage", {})
        return (
            f"finish_reason={finish_reason} "
            f"has_hidden_reasoning_field={has_reasoning_field} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"reasoning_preview={reasoning_preview!r}"
        )
    except Exception as e:
        return f"(could not diagnose: {e})"
 