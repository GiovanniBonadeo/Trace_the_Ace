"""
run_eval.py

Renders one selected prompt template for N sampled sessions, calls your
local OpenAI-compatible server, and saves results to CSV files for
scoring against train_labels.csv.

Adds (based on professor's working approach):
  - Retry-with-backoff on any parse/validation failure, not just network errors
  - Strict validation (bad JSON / out-of-range probability / etc. triggers a retry)
  - Resumable checkpointing: each result is written immediately to a .jsonl file;
    re-running the script skips already-completed responses for the same prompt
  - Automatic metrics (log loss, Brier score, ROC AUC, calibration error)

USAGE
-----
1. Edit the CONFIG block below (file paths, PROMPT_NAME, N_SAMPLES).
2. Run:  python run_eval.py
3. To just see available prompt names without running anything:
       python run_eval.py --list-prompts
4. If you interrupt the run (Ctrl+C) or it crashes, just run it again with the
   same PROMPT_NAME -- already-completed responses are skipped automatically.
"""

import os
import re
import sys
import json
import time
import math
import hashlib
import argparse
import requests
import numpy as np
import pandas as pd

# ============================================================
# CONFIG — edit these
# ============================================================

PROMPTS_DIR = "prompts"  # directory containing one .txt file per template, e.g. prompts/D8.txt
FEATURES_PATH = "train_features.csv"
LABELS_PATH = "train_labels.csv"
TRANSCRIPTS_DIR = "train_transcripts"  # directory containing one file per session_id

# Pick which template to run. Run with --list-prompts to see all names.
PROMPT_NAME = "ABCD4"

N_SAMPLES = 100
RANDOM_SEED = 20260721  # fixed seed so the same 100 sessions are picked every run
STRATIFY_BY_LABEL = True  # if True, sample ~50% is_correct=1 and ~50% is_correct=0

# retry behaviour: any parse/validation failure gets retried, not just network errors
RETRIES = 2                 # number of *extra* attempts after the first (total attempts = RETRIES + 1)
RETRY_BACKOFF_SECONDS = 2   # doubles each attempt, capped at 10s

OUTPUT_CSV = f"results_{PROMPT_NAME}.csv"                  # full debug output (raw response, timing, errors)
SUBMISSION_CSV = f"submission_{PROMPT_NAME}.csv"            # competition format: response_id,probability
METRICS_CSV = f"metrics_{PROMPT_NAME}.csv"                  # log loss / brier / auc / calibration
RENDERED_DIR = f"rendered_prompts_{PROMPT_NAME}"             # saved .txt of what was actually sent
CHECKPOINT_JSONL = f"checkpoint_{PROMPT_NAME}.jsonl"         # incremental results, enables resuming

# probability used ONLY when scoring metrics for a response with no valid probability
# (the raw None is still preserved in the checkpoint/results CSV for debugging)
FALLBACK_PROBABILITY = 0.5

# --- server config (same pattern as your working call) ---
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://10.70.13.33:11434")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "sk-RZSBTkuZYOeXULKBTKupkA")
MODEL_NAME = "qwen3.5-35b-a3b"
TEMPERATURE = 0.0
MAX_TOKENS = 3000
ENABLE_THINKING = False
REQUEST_TIMEOUT = 120  # seconds, per call


# ============================================================
# 1. Load prompt templates: one .txt file per template in PROMPTS_DIR
# ============================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt_templates(prompts_dir: str) -> dict:
    templates = {}
    for fname in sorted(os.listdir(prompts_dir)):
        if fname.endswith(".txt"):
            name = fname[:-4]  # strip ".txt"
            name = re.sub(r"[_\-]?prompt$", "", name, flags=re.IGNORECASE)
            with open(os.path.join(prompts_dir, fname), encoding="utf-8") as f:
                templates[name] = f.read().strip()
    return templates


# ============================================================
# 2. Build conversation text for a given session_id
# ============================================================

EXPECTED_TRANSCRIPT_COLS = {"session_id", "utterance_id", "role", "content", "timestamp"}


def find_session_file(transcripts_dir: str, session_id: str) -> str:
    for ext in (".csv", ".txt", ""):
        candidate = os.path.join(transcripts_dir, f"{session_id}{ext}")
        if os.path.isfile(candidate):
            return candidate
    import glob
    matches = glob.glob(os.path.join(transcripts_dir, f"{session_id}*"))
    if matches:
        return matches[0]
    return None


def build_conversation_text(transcripts_dir: str, session_id: str) -> str:
    path = find_session_file(transcripts_dir, session_id)
    if path is None:
        return ""

    df = pd.read_csv(path)
    missing = EXPECTED_TRANSCRIPT_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Transcript file {path} is missing columns: {missing}")

    df = df.sort_values("utterance_id")
    lines = []
    for _, row in df.iterrows():
        lines.append(f"{row['role']}: {row['content']}  [{row['timestamp']}]")
    return "\n".join(lines)


# ============================================================
# 2b. Sample N responses, optionally stratified 50/50 by is_correct
# ============================================================

def sample_responses(features: pd.DataFrame, labels_path: str, n_samples: int,
                      random_state: int, stratify: bool) -> pd.DataFrame:
    if not stratify:
        return features.sample(n=min(n_samples, len(features)), random_state=random_state).reset_index(drop=True)

    labels = pd.read_csv(labels_path)
    merged = features.merge(labels, on="response_id", how="inner")

    n_missing_labels = len(features) - len(merged)
    if n_missing_labels:
        print(f"NOTE: {n_missing_labels} responses had no matching label and were excluded from stratified sampling.")

    correct = merged[merged["is_correct"] == 1.0]
    incorrect = merged[merged["is_correct"] == 0.0]

    n_each = n_samples // 2
    n_correct = min(n_each, len(correct))
    n_incorrect = min(n_samples - n_correct, len(incorrect))

    if n_correct < n_each or n_incorrect < n_each:
        print(f"NOTE: requested {n_each}/{n_each} split but only {len(correct)} correct "
              f"and {len(incorrect)} incorrect rows available with labels. "
              f"Using {n_correct} correct + {n_incorrect} incorrect.")

    sample = pd.concat([
        correct.sample(n=n_correct, random_state=random_state),
        incorrect.sample(n=n_incorrect, random_state=random_state),
    ]).sample(frac=1, random_state=random_state)

    return sample.drop(columns=["is_correct"]).reset_index(drop=True)


# ============================================================
# 3. Render a template with {{topic}} and {{conversation}}
# ============================================================

def render_prompt(template: str, topic: str, conversation: str) -> str:
    text = template.replace("{{topic}}", topic)
    text = text.replace("{{conversation}}", conversation)
    return text


# ============================================================
# 4. Call the local server
# ============================================================

def call_server(user_prompt: str, seed: int) -> dict:
    base = LOCAL_API_URL.rstrip("/")
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"

    messages = [
        {"role": "system", "content": "Follow the requested JSON output exactly."},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }
    headers = {"Content-Type": "application/json"}
    if LOCAL_API_KEY and LOCAL_API_KEY != "your-local-api-key":
        headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"

    response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


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


# ============================================================
# 5. Strict JSON parsing + validation (raises on failure -> triggers retry)
# ============================================================

def parse_json_object(content: str) -> dict:
    """Extract and parse the model's JSON object. Raises ValueError if none found/invalid."""
    text = content.strip()

    # strip <think>...</think> blocks some models emit even with thinking disabled
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # strip markdown code fences if present
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")

    parsed = json.loads(text[start:end + 1])  # raises json.JSONDecodeError if malformed
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON is not an object")
    return parsed


def normalize_probability(value) -> float | None:
    """None is a legitimate value (model explicitly said 'not enough info').
    Anything else must be a finite number in [0, 1], or this raises -> triggers retry."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean probability is invalid")
    probability = float(value)
    if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability outside [0,1]: {value!r}")
    return probability


def validate_and_extract_probability(content: str) -> float | None:
    """Raises ValueError/json.JSONDecodeError on any malformed output -> caller retries."""
    parsed = parse_json_object(content)
    achievement = parsed.get("achievement")
    if not isinstance(achievement, dict) or "probability" not in achievement:
        raise ValueError("Missing achievement.probability in JSON")
    return normalize_probability(achievement["probability"])


# ============================================================
# 6. Call server with retries: any failure (network OR validation) is retried
# ============================================================

def call_with_retries(rendered_prompt: str, base_seed: int) -> dict:
    """Returns a result dict with keys: probability, raw_content, error, elapsed_sec, attempts."""
    last_error = None
    last_content = None

    for attempt in range(RETRIES + 1):
        t0 = time.time()
        try:
            server_response = call_server(rendered_prompt, seed=base_seed + attempt)
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

        if attempt < RETRIES:
            sleep_time = min(RETRY_BACKOFF_SECONDS * (2 ** attempt), 10)
            print(f"    attempt {attempt + 1} failed ({last_error}); retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    return {
        "probability": None,
        "raw_content": last_content,
        "error": last_error,
        "elapsed_sec": round(time.time() - t0, 2),
        "attempts": RETRIES + 1,
    }


# ============================================================
# 7. Checkpointing (resumable runs)
# ============================================================

def append_checkpoint(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_completed(path: str, prompt_hash: str) -> dict:
    """Returns {response_id: record} for entries matching the current prompt's hash."""
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("prompt_hash") == prompt_hash:
                completed[record["response_id"]] = record
    return completed


# ============================================================
# 8. Metrics
# ============================================================

def compute_metrics(results_df: pd.DataFrame, labels_path: str, fallback: float) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    merged = results_df.merge(labels, on="response_id", how="inner")

    if merged.empty:
        print("WARNING: no results matched labels; skipping metrics.")
        return pd.DataFrame()

    y = merged["is_correct"].astype(float).to_numpy()
    p = pd.to_numeric(merged["probability"], errors="coerce").fillna(fallback).astype(float).to_numpy()
    eps = 1e-6
    p_clipped = np.clip(p, eps, 1 - eps)

    log_loss = float(-(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)).mean())
    brier = float(np.mean((p_clipped - y) ** 2))

    # ROC AUC via rank-sum (no sklearn dependency)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos > 0 and n_neg > 0:
        ranks = pd.Series(p_clipped).rank(method="average").to_numpy()
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = float("nan")

    # expected calibration error, 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_clipped >= lo) & (p_clipped <= hi if i == 9 else p_clipped < hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p_clipped[mask].mean())

    metrics = pd.DataFrame([{
        "prompt_name": PROMPT_NAME,
        "n": len(merged),
        "log_loss": log_loss,
        "brier_score": brier,
        "roc_auc": auc,
        "ece_10_bins": float(ece),
        "mean_probability": float(p_clipped.mean()),
        "label_prevalence": float(y.mean()),
        "parse_success_rate": float(merged["probability"].notna().mean()),
        "mean_elapsed_sec": float(merged["elapsed_sec"].mean()) if "elapsed_sec" in merged else None,
        "mean_attempts": float(merged["attempts"].mean()) if "attempts" in merged else None,
    }])
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-prompts", action="store_true", help="List available prompt names and exit")
    args = parser.parse_args()

    templates = load_prompt_templates(PROMPTS_DIR)

    if args.list_prompts:
        print("Available prompt names:")
        for name in templates:
            print(f"  - {name}")
        return

    if PROMPT_NAME not in templates:
        print(f"ERROR: '{PROMPT_NAME}' not found. Available: {list(templates.keys())}")
        sys.exit(1)

    template = templates[PROMPT_NAME]
    prompt_hash = sha256_text(template)

    features = pd.read_csv(FEATURES_PATH)

    if not os.path.isdir(TRANSCRIPTS_DIR):
        print(f"ERROR: TRANSCRIPTS_DIR '{TRANSCRIPTS_DIR}' does not exist or is not a directory.")
        sys.exit(1)

    sample = sample_responses(features, LABELS_PATH, N_SAMPLES, RANDOM_SEED, STRATIFY_BY_LABEL)

    os.makedirs(RENDERED_DIR, exist_ok=True)

    # resume: load already-completed responses for THIS prompt's exact text (hash-matched)
    completed = load_completed(CHECKPOINT_JSONL, prompt_hash)
    if completed:
        print(f"[RESUME] Found {len(completed)} already-completed responses for this prompt version; skipping them.")

    for i, row in sample.iterrows():
        response_id = row["response_id"]
        session_id = row["session_id"]
        topic = row["learning_objective"]

        if response_id in completed:
            print(f"[{i+1}/{len(sample)}] {response_id}: already done, skipping")
            continue

        conversation = build_conversation_text(TRANSCRIPTS_DIR, session_id)
        if not conversation:
            print(f"[{i+1}/{len(sample)}] {response_id}: WARNING no transcript file found for session {session_id}, skipping")
            continue

        rendered = render_prompt(template, topic, conversation)

        with open(os.path.join(RENDERED_DIR, f"{response_id}.txt"), "w", encoding="utf-8") as f:
            f.write(rendered)

        print(f"[{i+1}/{len(sample)}] {response_id} (session {session_id}) -> calling server...")
        result = call_with_retries(rendered, base_seed=RANDOM_SEED)

        print(f"  done in {result['elapsed_sec']}s after {result['attempts']} attempt(s), "
              f"probability={result['probability']}")
        if result["probability"] is None:
            preview = (result["raw_content"] or "")[:300].replace("\n", " ")
            print(f"  FAILED after all retries. error={result['error']} | preview: {preview}")

        record = {
            "response_id": response_id,
            "session_id": session_id,
            "prompt_name": PROMPT_NAME,
            "prompt_hash": prompt_hash,
            "probability": result["probability"],
            "raw_content": result["raw_content"],
            "error": result["error"],
            "elapsed_sec": result["elapsed_sec"],
            "attempts": result["attempts"],
        }
        append_checkpoint(CHECKPOINT_JSONL, record)
        completed[response_id] = record

    # build final outputs from ALL completed records for this prompt hash
    # (covers both this run and any prior resumed runs)
    sample_ids = set(sample["response_id"])
    final_records = [r for rid, r in completed.items() if rid in sample_ids]
    out_df = pd.DataFrame(final_records)

    if out_df.empty:
        print("\nWARNING: no results were produced (every row may have been skipped, "
              "e.g. missing transcript files). Nothing to save.")
        return

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} detailed results to {OUTPUT_CSV}")
    print(f"Rendered prompts saved under {RENDERED_DIR}/")
    print(f"Checkpoint (resumable) saved to {CHECKPOINT_JSONL}")

    n_missing = out_df["probability"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} response(s) had no parseable probability after retries; "
              f"filling with FALLBACK_PROBABILITY={FALLBACK_PROBABILITY} in submission file")

    submission_df = out_df[["response_id", "probability"]].copy()
    submission_df["probability"] = pd.to_numeric(submission_df["probability"], errors="coerce").fillna(FALLBACK_PROBABILITY)
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Saved submission file ({len(submission_df)} rows) to {SUBMISSION_CSV}")

    metrics_df = compute_metrics(out_df, LABELS_PATH, FALLBACK_PROBABILITY)
    if not metrics_df.empty:
        metrics_df.to_csv(METRICS_CSV, index=False)
        print(f"Saved metrics to {METRICS_CSV}")
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
