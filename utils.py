import os
import json
import hashlib
import re
import pandas as pd

# Load prompt templates: one .txt file per template in PROMPTS_DIR
 
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
  
# Sample N responses, optionally stratified 50/50 by "is_correct" variable
 
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
 
 
# Render a template with {{topic}} and {{conversation}}
 
def render_prompt(template: str, topic: str, conversation: str) -> str:
    text = template.replace("{{topic}}", topic)
    text = text.replace("{{conversation}}", conversation)
    return text
 
# Checkpointing (resumable runs)
 
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
 
