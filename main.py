"""
Renders one selected prompt template (from prompts.txt) for N sampled
sessions, calls your local OpenAI-compatible server, and saves the raw +
parsed results to a CSV for scoring against train_labels.csv.

USAGE
-----
1. Edit the CONFIG block below (file paths, PROMPT_NAME, N_SAMPLES).
2. Run:  python run_eval.py
3. To just see available prompt names without running anything:
       python main.py --list-prompts
"""

import os
import re
import sys
import json
import time
import argparse
import requests 
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROMPTS_DIR = "prompts"  # directory containing one .txt file per template, e.g. prompts/D8.txt
FEATURES_PATH = "train_features.csv"
LABELS_PATH = "train_labels.csv"
TRANSCRIPTS_DIR = "train_transcripts" 

# Pick which template to run. Run with --list-prompts to see all names.
PROMPT_NAME = "ABCD4"

N_SAMPLES = 1
RANDOM_SEED = 20260721  # fixed seed so the same 100 sessions are picked every run
STRATIFY_BY_LABEL = True # if True, sample ~50% is_correct=1 and ~50% is_correct=0

OUTPUT_CSV = f"results_{PROMPT_NAME}.csv"                 # full debug output (raw response, timing, errors)
SUBMISSION_CSV = f"submission_{PROMPT_NAME}.csv"           # competition format: response_id,probability
RENDERED_DIR = f"rendered_prompts_{PROMPT_NAME}"           # saved .txt of what was actually sent

# probability used when a call fails or the model output can't be parsed,
# so the submission file always has a value for every response_id
FALLBACK_PROBABILITY = 0.5

# --- server config ---
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://10.70.13.33:11434")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "sk-RZSBTkuZYOeXULKBTKupkA")
MODEL_NAME = "qwen3.5-256k"
TEMPERATURE = 0.0
MAX_TOKENS = 2500
ENABLE_THINKING = False
REQUEST_TIMEOUT = 120  # seconds, per call

# ============================================================
# 1. Load prompt templates: one .txt file per template in PROMPTS_DIR
#    e.g. prompts/D8.txt, prompts/ABCD4.txt, prompts/A4.txt ...
# ============================================================

def load_prompt_templates(prompts_dir: str) -> dict:
    templates = {}
    for fname in sorted(os.listdir(prompts_dir)):
        if fname.endswith(".txt"):
            name = fname[:-4]  # strip ".txt"
            with open(os.path.join(prompts_dir, fname), encoding="utf-8") as f:
                templates[name] = f.read().strip()
    return templates


# ============================================================
# 2. Build conversation text for a given session_id
#    (one file per session, inside TRANSCRIPTS_DIR)
# ============================================================

EXPECTED_TRANSCRIPT_COLS = {"session_id", "utterance_id", "role", "content", "timestamp"}


def find_session_file(transcripts_dir: str, session_id: str) -> str:
    """Locate the transcript file for a session_id, trying common extensions."""
    for ext in (".csv", ".txt", ""):
        candidate = os.path.join(transcripts_dir, f"{session_id}{ext}")
        if os.path.isfile(candidate):
            return candidate
    # fallback: glob for anything starting with session_id
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
    ]).sample(frac=1, random_state=random_state)  # shuffle so correct/incorrect aren't grouped in run order

    return sample.drop(columns=["is_correct"]).reset_index(drop=True)


# ============================================================
# 3. Render a template with {{topic}} and {{conversation}}
# ============================================================

def render_prompt(template: str, topic: str, conversation: str) -> str:
    text = template.replace("{{topic}}", topic)
    text = text.replace("{{conversation}}", conversation)
    return text


# ============================================================
# 4. Call the local server (your exact pattern, wrapped in a function)
# ============================================================

def call_server(user_prompt: str) -> dict:
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
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }
    headers = {"Content-Type": "application/json"}
    if LOCAL_API_KEY and LOCAL_API_KEY != "sk-RZSBTkuZYOeXULKBTKupkA":
        headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"

    response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def extract_content(server_response: dict) -> str:
    try:
        return server_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def extract_probability(content: str):
    """Best-effort parse of the model's JSON output to pull out achievement.probability."""
    text = content.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    try:
        parsed = json.loads(text)
        return parsed.get("achievement", {}).get("probability", None), parsed
    except json.JSONDecodeError:
        return None, None


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

    features = pd.read_csv(FEATURES_PATH)

    if not os.path.isdir(TRANSCRIPTS_DIR):
        print(f"ERROR: TRANSCRIPTS_DIR '{TRANSCRIPTS_DIR}' does not exist or is not a directory.")
        sys.exit(1)

    # sample N responses (fixed seed -> reproducible sample; optionally stratified 50/50 by outcome)
    sample = sample_responses(features, LABELS_PATH, N_SAMPLES, RANDOM_SEED, STRATIFY_BY_LABEL)

    os.makedirs(RENDERED_DIR, exist_ok=True)
    results = []

    for i, row in sample.iterrows():
        response_id = row["response_id"]
        session_id = row["session_id"]
        topic = row["learning_objective"]

        conversation = build_conversation_text(TRANSCRIPTS_DIR, session_id)
        if not conversation:
            print(f"[{i+1}/{len(sample)}] {response_id}: WARNING no transcript file found for session {session_id}, skipping")
            continue

        rendered = render_prompt(template, topic, conversation)

        # save exactly what was sent, for spot-checking (mirrors professor's rendered_prompt_examples/)
        with open(os.path.join(RENDERED_DIR, f"{response_id}.txt"), "w", encoding="utf-8") as f:
            f.write(rendered)

        print(f"[{i+1}/{len(sample)}] {response_id} (session {session_id}) -> calling server...")
        t0 = time.time()
        try:
            server_response = call_server(rendered)
        except requests.exceptions.RequestException as e:
            print(f"  ERROR calling server: {e}")
            results.append({
                "response_id": response_id, "session_id": session_id,
                "prompt_name": PROMPT_NAME, "probability": None,
                "raw_content": None, "error": str(e), "elapsed_sec": time.time() - t0,
            })
            continue

        elapsed = time.time() - t0
        content = extract_content(server_response)
        probability, parsed_json = extract_probability(content)

        print(f"  done in {elapsed:.1f}s, probability={probability}")

        results.append({
            "response_id": response_id,
            "session_id": session_id,
            "prompt_name": PROMPT_NAME,
            "probability": probability,
            "raw_content": content,
            "error": None,
            "elapsed_sec": round(elapsed, 2),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} detailed results to {OUTPUT_CSV}")
    print(f"Rendered prompts saved under {RENDERED_DIR}/")

    # --- competition-format submission file: response_id,probability ---
    n_missing = out_df["probability"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} response(s) had no parseable probability; "
              f"filling with FALLBACK_PROBABILITY={FALLBACK_PROBABILITY}")

    submission_df = out_df[["response_id", "probability"]].copy()
    submission_df["probability"] = submission_df["probability"].fillna(FALLBACK_PROBABILITY)
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Saved submission file ({len(submission_df)} rows) to {SUBMISSION_CSV}")


if __name__ == "__main__":
    main()
