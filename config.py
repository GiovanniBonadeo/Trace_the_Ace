import os

PROMPTS_DIR = "prompts"  # directory containing one .txt file per template, e.g. prompts/D8.txt
FEATURES_PATH = "train_features.csv"
LABELS_PATH = "train_labels.csv"
TRANSCRIPTS_DIR = "train_transcripts"  # directory containing one file per session_id
 
# Pick which template to run. Run with --list-prompts to see all names.
PROMPT_NAME = "A4"
 
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
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "CENSORED")
MODEL_NAME = "qwen3.5-32k"
TEMPERATURE = 0.0
MAX_TOKENS = 3000
ENABLE_THINKING = False
REQUEST_TIMEOUT = 120  # seconds, per call
 
EXPECTED_TRANSCRIPT_COLS = {"session_id", "utterance_id", "role", "content", "timestamp"}
 