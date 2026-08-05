# Building conversations from a session_id utils 

import os
import pandas as pd
from config import *
 
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
 