"""""
 
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
import sys
import argparse
import pandas as pd

from config import *
from call_server import *
from build_conv import *
from metrics import *
from utils import *
 
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

        result = call_with_retries(rendered, base_seed=RANDOM_SEED, retries=RETRIES, retry_backoff_seconds=RETRY_BACKOFF_SECONDS, local_url=LOCAL_API_URL, local_key=LOCAL_API_KEY, model_name=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, enable_thinking=ENABLE_THINKING, request_timeout=REQUEST_TIMEOUT )
 
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
 