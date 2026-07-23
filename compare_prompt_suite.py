"""
Compare learner-modelling prompts on a balanced labelled training sample.

The model predicts whether the learner will answer the next assessment question
for the supplied learning objective correctly.

Guarantees:
- Every prompt receives the complete transcript.
- Transcript rows are never filtered, truncated, summarized, or reordered.
- Training labels are used only for sampling and evaluation.
- Both `correct` and `is_correct` label columns are accepted.
- Predictions are saved incrementally and interrupted runs can resume.
- Prompt file hashes are stored, so changed prompts are rerun automatically.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from urllib import error, request

import numpy as np
import pandas as pd


LABEL_SUFFIX_RE = re.compile(
    r"\s*(?:—|-|:)?\s*(?:is_correct|correct)\s*=\s*(?:0(?:\.0)?|1(?:\.0)?)\s*$",
    flags=re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclasses.dataclass(frozen=True)
class PromptSpec:
    id: str
    title: str
    path: Path
    feature_count: int
    family: str
    dimension_codes: tuple[str, ...]
    label_stems: dict[str, str]
    schema_version: int
    order: int
    prompt_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to train_features.csv")
    parser.add_argument("--labels", required=True, help="Path to train_labels.csv")
    parser.add_argument(
        "--transcripts-dir",
        required=True,
        help="Directory containing one {session_id}.csv per session",
    )
    parser.add_argument(
        "--prompt-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing prompt_manifest.json and prompts/",
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--length-bins", type=int, default=4)
    parser.add_argument("--max-per-session", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)

    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "lm-studio"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--thinking-mode",
        choices=["off", "on", "both", "default"],
        default="off",
    )
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="Extra JSON merged into each chat-completions payload",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Optional prompt IDs; default is every prompt in manifest order",
    )
    parser.add_argument(
        "--null-fallback",
        default="prevalence",
        help="Metric-only fallback for null/error predictions: prevalence, 0.5, etc.",
    )
    parser.add_argument("--clip-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def sanitize_topic(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    previous = None
    while previous != text:
        previous = text
        text = LABEL_SUFFIX_RE.sub("", text).strip()
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt_specs(
    prompt_dir: Path,
    selected: list[str] | None,
) -> list[PromptSpec]:
    records = json.loads(
        (prompt_dir / "prompt_manifest.json").read_text(encoding="utf-8")
    )

    specs: list[PromptSpec] = []
    for record in records:
        path = prompt_dir / str(record["path"])
        template = path.read_text(encoding="utf-8")
        specs.append(
            PromptSpec(
                id=str(record["id"]),
                title=str(record["title"]),
                path=path,
                feature_count=int(record["feature_count"]),
                family=str(record["family"]),
                dimension_codes=tuple(
                    str(code) for code in record.get("dimension_codes", [])
                ),
                label_stems={
                    str(code): str(stem)
                    for code, stem in record.get("label_stems", {}).items()
                },
                schema_version=int(record.get("schema_version", 1)),
                order=int(record["order"]),
                prompt_hash=sha256_text(template),
            )
        )

    specs.sort(key=lambda spec: (spec.feature_count, spec.order))

    if selected:
        wanted = set(selected)
        unknown = wanted - {spec.id for spec in specs}
        if unknown:
            raise ValueError(f"Unknown prompt IDs: {sorted(unknown)}")
        specs = [spec for spec in specs if spec.id in wanted]

    return specs


def transcript_path(directory: Path, session_id: str) -> Path:
    return directory / f"{session_id}.csv"


def read_full_transcript(path: Path) -> tuple[str, int, int]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = ["session_id", "utterance_id", "role", "content", "timestamp"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing transcript columns {missing}")

    # Preserve all rows and any additional columns.
    extra = [column for column in frame.columns if column not in required]
    frame = frame[required + extra]
    text = frame.to_csv(index=False, lineterminator="\n")
    return text, len(text), len(frame)


def add_transcript_metadata(
    frame: pd.DataFrame,
    transcripts_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    text_cache: dict[str, str] = {}
    char_cache: dict[str, int] = {}
    turn_cache: dict[str, int] = {}

    for session_id in frame["session_id"].astype(str).unique():
        path = transcript_path(transcripts_dir, session_id)
        if not path.exists():
            raise FileNotFoundError(path)
        text, chars, turns = read_full_transcript(path)
        text_cache[session_id] = text
        char_cache[session_id] = chars
        turn_cache[session_id] = turns

    output = frame.copy()
    output["transcript_chars"] = output["session_id"].astype(str).map(char_cache)
    output["turn_count"] = output["session_id"].astype(str).map(turn_cache)
    return output, text_cache


def assign_length_bins(
    frame: pd.DataFrame,
    requested_bins: int,
) -> pd.DataFrame:
    output = frame.copy()
    unique_lengths = int(output["transcript_chars"].nunique())
    bin_count = max(1, min(requested_bins, unique_lengths))

    if bin_count == 1:
        output["length_bin"] = 0
    else:
        output["length_bin"] = pd.qcut(
            output["transcript_chars"],
            q=bin_count,
            labels=False,
            duplicates="drop",
        ).astype(int)
    return output


def balanced_sample(
    frame: pd.DataFrame,
    sample_size: int,
    length_bins: int,
    max_per_session: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("--sample-size must be positive")
    if max_per_session < 1:
        raise ValueError("--max-per-session must be positive")

    observed_labels = set(frame["correct"].dropna().astype(float).unique())
    if not observed_labels.issubset({0.0, 1.0}):
        raise ValueError("Labels must contain only 0.0 and 1.0")

    work = assign_length_bins(frame, length_bins)
    rng = random.Random(seed)
    bins = sorted(work["length_bin"].astype(int).unique())

    targets = {
        0.0: sample_size // 2,
        1.0: sample_size - sample_size // 2,
    }

    selected: list[int] = []
    session_counts: dict[str, int] = {}

    def can_add(index: int) -> bool:
        session_id = str(work.at[index, "session_id"])
        return session_counts.get(session_id, 0) < max_per_session

    def add(index: int) -> None:
        selected.append(index)
        session_id = str(work.at[index, "session_id"])
        session_counts[session_id] = session_counts.get(session_id, 0) + 1

    for label in (0.0, 1.0):
        target = targets[label]
        base, remainder = divmod(target, max(1, len(bins)))

        for position, length_bin in enumerate(bins):
            requested = base + (1 if position < remainder else 0)
            candidates = work.index[
                (work["correct"].astype(float) == label)
                & (work["length_bin"].astype(int) == length_bin)
            ].tolist()
            rng.shuffle(candidates)

            taken = 0
            for index in candidates:
                if can_add(index):
                    add(index)
                    taken += 1
                if taken >= requested:
                    break

        current = sum(
            float(work.at[index, "correct"]) == label for index in selected
        )
        remaining = work.index[
            (work["correct"].astype(float) == label)
            & (~work.index.isin(selected))
        ].tolist()
        rng.shuffle(remaining)

        for index in remaining:
            if current >= target:
                break
            if can_add(index):
                add(index)
                current += 1

    remaining = work.index[~work.index.isin(selected)].tolist()
    rng.shuffle(remaining)
    for index in remaining:
        if len(selected) >= sample_size:
            break
        if can_add(index):
            add(index)

    sample = work.loc[selected].copy()
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(sample) < sample_size:
        print(
            f"WARNING: selected {len(sample)} of {sample_size} responses under "
            f"max_per_session={max_per_session}",
            file=sys.stderr,
        )

    return sample


def render_prompt(
    template: str,
    topic: Any,
    conversation: str,
) -> str:
    clean_topic = sanitize_topic(topic)
    if re.search(
        r"(?:is_correct|correct)\s*=\s*[01](?:\.0)?",
        clean_topic,
        re.IGNORECASE,
    ):
        raise ValueError("Correctness metadata remains in the learning objective")
    return (
        template
        .replace("{{topic}}", clean_topic)
        .replace("{{conversation}}", conversation)
    )


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = THINK_RE.sub("", text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found")
        value = json.loads(cleaned[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("Top-level JSON must be an object")
    return value


def normalize_probability(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean probability is invalid")

    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability outside [0,1]: {value!r}")
    return probability


QUALITY_WORDS = {
    0: "minimal",
    1: "weak",
    2: "mixed",
    3: "strong",
    4: "exceptional",
}


SCORE_ADJECTIVES: dict[int, str] = {
    0: "minimal",
    1: "weak",
    2: "mixed",
    3: "strong",
    4: "exceptional",
}

DIMENSION_LABEL_STEMS: dict[str, str] = {
    # A — Direct mastery evidence
    "ES": "final accuracy",
    "IRV": "reasoning validity",
    "LGC": "objective coverage",
    "NTP": "transfer performance",
    "COC": "performance consistency",
    "SMR": "misconception resolution",
    "SEQ": "self explanation",
    "PC": "procedural completeness",
    "SSA": "strategy selection",
    "SEC": "error correction",
    "WCR": "knowledge retention",
    "RP": "representation precision",
    "JEU": "evidence justification",
    "BCU": "boundary understanding",
    "GPA": "generative ability",
    "CC": "confidence calibration",

    # B — Evidence conditions
    "SLD": "independent demonstration",
    "TAC": "answer independence",
    "LOA": "objective alignment",
    "IEO": "independent opportunities",
    "ED": "evidence diversity",
    "ECL": "challenge adequacy",
    "PIV": "independent verification",
    "ER": "evidence recency",
    "QD": "question diagnosticity",
    "GCR": "response authenticity",
    "RCO": "response opportunity",
    "PCI": "prompt clarity",
    "SRR": "speaker reliability",
    "RO": "response observability",
    "IER": "example independence",
    "OCB": "component coverage",

    # C — Learning process
    "PIT": "improvement trajectory",
    "SFS": "scaffold fading",
    "IFU": "feedback uptake",
    "FT": "feedback transfer",
    "ERR": "error reduction",
    "LQD": "question quality",
    "MM": "metacognitive monitoring",
    "SRF": "strategy revision",
    "PTD": "learning persistence",
    "CE": "constructive elaboration",
    "RRE": "retrieval effort",
    "TC": "teacher contingency",
    "TR": "learner responsibility",
    "FSE": "feedback quality",
    "PCL": "productive challenge",
    "ESR": "engagement recovery",

    # D — Learner/topic/system context
    "EKS": "entry knowledge",
    "PR": "prerequisite readiness",
    "EPE": "prior exposure",
    "LTD": "topic manageability",
    "CAL": "conceptual manageability",
    "PBL": "procedural manageability",
    "LCB": "language accessibility",
    "ELC": "expressive adequacy",
    "WMS": "memory manageability",
    "NCB": "notation manageability",
    "TFF": "format familiarity",
    "DGC": "developmental fit",
    "AL": "affective readiness",
    "MGO": "learning motivation",
    "RIE": "instructional exposure",
    "ESD": "environmental stability",
}


def canonical_dimension_label(
    code: str,
    score: int | None,
    label_stems: dict[str, str] | None = None,
) -> str:
    if score is None:
        return "not observable"

    stems = DIMENSION_LABEL_STEMS.copy()
    if label_stems:
        stems.update(
            {
                str(key): str(value).strip()
                for key, value in label_stems.items()
            }
        )

    stem = stems.get(code, code.lower())
    return f"{SCORE_ADJECTIVES[score]} {stem}"


def normalize_dimensions(
    value: Any,
    expected_codes: tuple[str, ...],
    label_stems: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Validate dimension scores and retain the LLM-generated labels.

    Exact label wording is audited but never used as a retry condition.
    """
    if not expected_codes:
        return {}
    if not isinstance(value, dict):
        raise ValueError("dimensions must be a JSON object")

    output: dict[str, dict[str, Any]] = {}

    for code in expected_codes:
        if code not in value:
            raise ValueError(f"{code}: missing dimension")

        raw_dimension = value[code]

        # Accept current nested output and historical numeric-only output.
        if isinstance(raw_dimension, dict):
            raw_score = raw_dimension.get("score")
            raw_model_label = raw_dimension.get("label")
        else:
            raw_score = raw_dimension
            raw_model_label = None

        if raw_score is None:
            score: int | None = None
        else:
            if isinstance(raw_score, bool):
                raise ValueError(f"{code}: boolean score is invalid")

            numeric_score = float(raw_score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"{code}: score is not finite")
            if not numeric_score.is_integer():
                raise ValueError(f"{code}: score must be an integer")

            score = int(numeric_score)
            if score not in {0, 1, 2, 3, 4}:
                raise ValueError(f"{code}: score outside 0-4")

        if raw_model_label is None:
            model_label: str | None = None
        elif isinstance(raw_model_label, str):
            model_label = " ".join(raw_model_label.strip().split())
        else:
            model_label = str(raw_model_label)

        canonical_label = canonical_dimension_label(
            code,
            score,
            label_stems=label_stems,
        )

        output[code] = {
            "score": score,
            "model_label": model_label,
            "canonical_label": canonical_label,
            "label_match": (
                model_label is not None
                and model_label.casefold() == canonical_label.casefold()
            ),
        }

    return output


def thinking_modes(mode: str) -> list[str]:
    return ["off", "on"] if mode == "both" else [mode]


def chat_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def call_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    seed: int,
    thinking_mode: str,
    extra_body: dict[str, Any],
) -> tuple[str, dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Follow the requested JSON output exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }

    if thinking_mode in {"on", "off"}:
        payload["chat_template_kwargs"] = {
            "enable_thinking": thinking_mode == "on"
        }

    payload.update(extra_body)

    request_object = request.Request(
        chat_endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with request.urlopen(request_object, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc

    elapsed = time.perf_counter() - started
    content = body["choices"][0]["message"].get("content") or ""
    return content, body.get("usage", {}), elapsed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def completed_keys(
    path: Path,
) -> set[tuple[str, str, str, str]]:
    completed: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return completed

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(
                    (
                        str(record["response_id"]),
                        str(record["prompt_id"]),
                        str(record["thinking_mode"]),
                        str(record.get("prompt_hash", "")),
                    )
                )
        except Exception:
            continue

    return completed


def run_request(
    *,
    row: dict[str, Any],
    spec: PromptSpec,
    template: str,
    transcript: str,
    args: argparse.Namespace,
    thinking_mode: str,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_prompt(template, row["learning_objective"], transcript)
    prompt_chars = len(prompt)
    started = time.perf_counter()
    last_error: str | None = None
    last_raw_output = ""
    last_usage: dict[str, Any] = {}
    last_elapsed: float | None = None

    for attempt in range(args.retries + 1):
        try:
            raw_output, usage, _ = call_model(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                seed=args.seed + attempt,
                thinking_mode=thinking_mode,
                extra_body=extra_body,
            )
            last_raw_output = raw_output
            last_usage = usage
            last_elapsed = _

            parsed = parse_json_object(raw_output)
            probability = normalize_probability(
                (parsed.get("achievement") or {}).get("probability")
            )
            dimensions = normalize_dimensions(
                parsed.get("dimensions", {}),
                spec.dimension_codes,
                spec.label_stems,
            )

            return {
                "status": "ok",
                "response_id": str(row["response_id"]),
                "session_id": str(row["session_id"]),
                "correct": float(row["correct"]),
                "learning_objective": sanitize_topic(row["learning_objective"]),
                "transcript_chars": int(row["transcript_chars"]),
                "turn_count": int(row["turn_count"]),
                "length_bin": int(row["length_bin"]),
                "prompt_id": spec.id,
                "prompt_title": spec.title,
                "prompt_hash": spec.prompt_hash,
                "schema_version": spec.schema_version,
                "feature_count": spec.feature_count,
                "family": spec.family,
                "thinking_mode": thinking_mode,
                "probability": probability,
                "dimensions": dimensions,
                "raw_output": raw_output,
                "usage": usage,
                "prompt_chars": prompt_chars,
                "output_chars": len(raw_output),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "elapsed_seconds": time.perf_counter() - started,
                "attempt_count": attempt + 1,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(min(2**attempt, 5))

    return {
        "status": "error",
        "response_id": str(row["response_id"]),
        "session_id": str(row["session_id"]),
        "correct": float(row["correct"]),
        "learning_objective": sanitize_topic(row["learning_objective"]),
        "transcript_chars": int(row["transcript_chars"]),
        "turn_count": int(row["turn_count"]),
        "length_bin": int(row["length_bin"]),
        "prompt_id": spec.id,
        "prompt_title": spec.title,
        "prompt_hash": spec.prompt_hash,
        "schema_version": spec.schema_version,
        "feature_count": spec.feature_count,
        "family": spec.family,
        "thinking_mode": thinking_mode,
        "probability": None,
        "dimensions": {},
        "raw_output": last_raw_output,
        "usage": last_usage,
        "prompt_chars": prompt_chars,
        "output_chars": len(last_raw_output),
        "prompt_tokens": last_usage.get("prompt_tokens"),
        "completion_tokens": last_usage.get("completion_tokens"),
        "total_tokens": last_usage.get("total_tokens"),
        "elapsed_seconds": time.perf_counter() - started,
        "attempt_count": args.retries + 1,
        "error": last_error,
    }


def fallback_probability(
    setting: str,
    prevalence: float,
) -> float:
    if setting.lower() == "prevalence":
        return float(prevalence)

    value = float(setting)
    if not 0.0 <= value <= 1.0:
        raise ValueError("--null-fallback must be prevalence or a number in [0,1]")
    return value


def roc_auc_binary(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2
        )
        / (positives * negatives)
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0

    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)

        if mask.any():
            value += float(mask.mean()) * abs(
                float(labels[mask].mean())
                - float(probabilities[mask].mean())
            )

    return float(value)


def latest_records(
    raw_path: Path,
    active_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}

    if not raw_path.exists():
        return []

    for line in raw_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            prompt_id = str(record["prompt_id"])
            if prompt_id not in active_hashes:
                continue
            if str(record.get("prompt_hash", "")) != active_hashes[prompt_id]:
                continue

            key = (
                str(record["response_id"]),
                prompt_id,
                str(record["thinking_mode"]),
            )
            latest[key] = record
        except Exception:
            continue

    return list(latest.values())


def write_aggregates(
    *,
    raw_path: Path,
    output_dir: Path,
    fallback: float,
    clip_epsilon: float,
    active_hashes: dict[str, str],
) -> None:
    records = latest_records(raw_path, active_hashes)
    if not records:
        return

    prediction_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for record in records:
        raw_probability = record.get("probability")
        scoring_probability = (
            fallback if raw_probability is None else float(raw_probability)
        )
        scoring_probability = min(
            1.0 - clip_epsilon,
            max(clip_epsilon, scoring_probability),
        )

        prompt_key = (
            f'{record["prompt_id"]}__thinking_{record["thinking_mode"]}'
        )

        prediction_rows.append(
            {
                "response_id": record["response_id"],
                "session_id": record["session_id"],
                "correct": record["correct"],
                "prompt_id": record["prompt_id"],
                "prompt_key": prompt_key,
                "prompt_hash": record.get("prompt_hash"),
                "schema_version": record.get("schema_version"),
                "thinking_mode": record["thinking_mode"],
                "feature_count": record["feature_count"],
                "family": record["family"],
                "transcript_chars": record["transcript_chars"],
                "turn_count": record["turn_count"],
                "length_bin": record["length_bin"],
                "status": record["status"],
                "raw_probability": raw_probability,
                "probability_for_scoring": scoring_probability,
                "used_fallback": raw_probability is None,
                "dimensions_json": json.dumps(
                    record.get("dimensions") or {},
                    ensure_ascii=False,
                ),
                "prompt_chars": record.get("prompt_chars"),
                "output_chars": record.get("output_chars"),
                "prompt_tokens": record.get("prompt_tokens"),
                "completion_tokens": record.get("completion_tokens"),
                "total_tokens": record.get("total_tokens"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "attempt_count": record.get("attempt_count"),
                "error": record.get("error"),
            }
        )

        for code, dimension in (record.get("dimensions") or {}).items():
            feature_rows.append(
                {
                    "response_id": record["response_id"],
                    "session_id": record["session_id"],
                    "correct": record["correct"],
                    "prompt_id": record["prompt_id"],
                    "prompt_key": prompt_key,
                    "prompt_hash": record.get("prompt_hash"),
                    "thinking_mode": record["thinking_mode"],
                    "feature_count": record["feature_count"],
                    "family": record["family"],
                    "dimension": code,
                    "score": dimension.get("score"),
                    "dimension_label": dimension.get("label"),
                    "raw_probability": raw_probability,
                    "probability_for_scoring": scoring_probability,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    features = pd.DataFrame(feature_rows)

    predictions.to_csv(output_dir / "predictions_long.csv", index=False)
    features.to_csv(output_dir / "feature_predictions_long.csv", index=False)

    prediction_wide = predictions.pivot_table(
        index=["response_id", "correct"],
        columns="prompt_key",
        values="probability_for_scoring",
        aggfunc="first",
    ).reset_index()
    prediction_wide.to_csv(
        output_dir / "predictions_wide.csv",
        index=False,
    )

    if not features.empty:
        features["feature_column"] = (
            features["prompt_key"].astype(str)
            + "__"
            + features["dimension"].astype(str)
        )

        score_wide = features.pivot_table(
            index=["response_id", "correct"],
            columns="feature_column",
            values="score",
            aggfunc="first",
        ).reset_index()
        score_wide.to_csv(
            output_dir / "feature_predictions_wide.csv",
            index=False,
        )

        label_wide = features.pivot_table(
            index=["response_id", "correct"],
            columns="feature_column",
            values="dimension_label",
            aggfunc="first",
        ).reset_index()
        label_wide.to_csv(
            output_dir / "feature_labels_wide.csv",
            index=False,
        )

    by_prompt = output_dir / "by_prompt"
    by_prompt.mkdir(exist_ok=True)
    for prompt_key, group in predictions.groupby("prompt_key", sort=False):
        group.to_csv(by_prompt / f"{prompt_key}.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    for prompt_key, group in predictions.groupby("prompt_key", sort=False):
        labels = group["correct"].to_numpy(dtype=float)
        probabilities = group["probability_for_scoring"].to_numpy(dtype=float)

        log_loss = float(
            -(
                labels * np.log(probabilities)
                + (1.0 - labels) * np.log(1.0 - probabilities)
            ).mean()
        )

        metric_rows.append(
            {
                "prompt_key": prompt_key,
                "prompt_id": group["prompt_id"].iloc[0],
                "thinking_mode": group["thinking_mode"].iloc[0],
                "feature_count": int(group["feature_count"].iloc[0]),
                "family": group["family"].iloc[0],
                "n": len(group),
                "log_loss": log_loss,
                "brier_score": float(
                    np.mean((probabilities - labels) ** 2)
                ),
                "roc_auc": roc_auc_binary(labels, probabilities),
                "ece_10_bins": expected_calibration_error(
                    labels,
                    probabilities,
                    bins=10,
                ),
                "mean_probability": float(probabilities.mean()),
                "label_prevalence": float(labels.mean()),
                "null_or_error_rate": float(group["used_fallback"].mean()),
                "parse_success_rate": float((group["status"] == "ok").mean()),
                "mean_elapsed_seconds": (
                    float(group["elapsed_seconds"].dropna().mean())
                    if group["elapsed_seconds"].notna().any()
                    else None
                ),
                "mean_prompt_chars": (
                    float(group["prompt_chars"].dropna().mean())
                    if group["prompt_chars"].notna().any()
                    else None
                ),
                "mean_output_chars": (
                    float(group["output_chars"].dropna().mean())
                    if group["output_chars"].notna().any()
                    else None
                ),
                "mean_prompt_tokens": (
                    float(group["prompt_tokens"].dropna().mean())
                    if group["prompt_tokens"].notna().any()
                    else None
                ),
                "mean_completion_tokens": (
                    float(group["completion_tokens"].dropna().mean())
                    if group["completion_tokens"].notna().any()
                    else None
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["log_loss", "feature_count", "prompt_key"]
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    if set(predictions["thinking_mode"].unique()) >= {"off", "on"}:
        paired = predictions.pivot_table(
            index=["response_id", "correct", "prompt_id"],
            columns="thinking_mode",
            values="probability_for_scoring",
            aggfunc="first",
        ).reset_index()

        if {"off", "on"}.issubset(paired.columns):
            paired["absolute_difference"] = (
                paired["on"] - paired["off"]
            ).abs()
            paired.to_csv(
                output_dir / "thinking_mode_comparison.csv",
                index=False,
            )


def format_number(
    value: Any,
    digits: int = 0,
) -> str:
    if value is None:
        return "na"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(numeric):
        return "na"
    if digits == 0:
        return str(int(round(numeric)))
    return f"{numeric:.{digits}f}"


def print_sample_result(
    number: int,
    total: int,
    record: dict[str, Any],
) -> None:
    probability = record.get("probability")
    probability_text = (
        "null"
        if probability is None
        else f"{float(probability):.4f}"
    )

    print(
        f"[SAMPLE {number}/{total}] "
        f"prompt={record['prompt_id']} "
        f"mode={record['thinking_mode']} "
        f"response={record['response_id']} "
        f"status={record['status']} "
        f"p={probability_text} "
        f"time={format_number(record.get('elapsed_seconds'), 2)}s "
        f"input={format_number(record.get('prompt_chars'))}chars/"
        f"{format_number(record.get('prompt_tokens'))}tok "
        f"output={format_number(record.get('output_chars'))}chars/"
        f"{format_number(record.get('completion_tokens'))}tok "
        f"attempts={record.get('attempt_count', 'na')}",
        flush=True,
    )

    if record.get("status") != "ok" and record.get("error"):
        print(
            f"  error={str(record['error'])[:500]}",
            flush=True,
        )


def current_prompt_records(
    *,
    raw_path: Path,
    spec: PromptSpec,
    thinking_mode: str,
    response_ids: set[str],
    active_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in latest_records(raw_path, active_hashes)
        if str(record.get("prompt_id")) == spec.id
        and str(record.get("prompt_hash")) == spec.prompt_hash
        and str(record.get("thinking_mode")) == thinking_mode
        and str(record.get("response_id")) in response_ids
    ]


def print_prompt_summary(
    *,
    raw_path: Path,
    output_dir: Path,
    spec: PromptSpec,
    thinking_mode: str,
    sample: pd.DataFrame,
    fallback: float,
    clip_epsilon: float,
    active_hashes: dict[str, str],
    new_requests: int,
    skipped_requests: int,
) -> None:
    records = current_prompt_records(
        raw_path=raw_path,
        spec=spec,
        thinking_mode=thinking_mode,
        response_ids=set(sample["response_id"].astype(str)),
        active_hashes=active_hashes,
    )

    if not records:
        print(
            f"[PROMPT END] prompt={spec.id} mode={thinking_mode} "
            f"records=0 new={new_requests} skipped={skipped_requests}",
            flush=True,
        )
        return

    labels = np.asarray(
        [float(record["correct"]) for record in records],
        dtype=float,
    )
    probabilities = np.asarray(
        [
            fallback
            if record.get("probability") is None
            else float(record["probability"])
            for record in records
        ],
        dtype=float,
    )
    probabilities = np.clip(
        probabilities,
        clip_epsilon,
        1.0 - clip_epsilon,
    )

    log_loss = float(
        -(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        ).mean()
    )

    ok_count = sum(record.get("status") == "ok" for record in records)
    error_count = len(records) - ok_count
    null_count = sum(record.get("probability") is None for record in records)

    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if record.get("elapsed_seconds") is not None
    ]
    input_chars = [
        float(record["prompt_chars"])
        for record in records
        if record.get("prompt_chars") is not None
    ]
    output_chars = [
        float(record["output_chars"])
        for record in records
        if record.get("output_chars") is not None
    ]
    input_tokens = [
        float(record["prompt_tokens"])
        for record in records
        if record.get("prompt_tokens") is not None
    ]
    output_tokens = [
        float(record["completion_tokens"])
        for record in records
        if record.get("completion_tokens") is not None
    ]

    print(
        f"[PROMPT END] prompt={spec.id} mode={thinking_mode} "
        f"n={len(records)}/{len(sample)} "
        f"new={new_requests} skipped={skipped_requests} "
        f"ok={ok_count} errors={error_count} null={null_count} "
        f"log_loss={log_loss:.6f} "
        f"mean_time={format_number(np.mean(elapsed) if elapsed else None, 2)}s "
        f"total_time={format_number(np.sum(elapsed) if elapsed else None, 2)}s "
        f"mean_input={format_number(np.mean(input_chars) if input_chars else None)}chars/"
        f"{format_number(np.mean(input_tokens) if input_tokens else None)}tok "
        f"mean_output={format_number(np.mean(output_chars) if output_chars else None)}chars/"
        f"{format_number(np.mean(output_tokens) if output_tokens else None)}tok",
        flush=True,
    )

    write_aggregates(
        raw_path=raw_path,
        output_dir=output_dir,
        fallback=fallback,
        clip_epsilon=clip_epsilon,
        active_hashes=active_hashes,
    )



def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "prompt"


def build_request_payload(
    *,
    args: argparse.Namespace,
    prompt: str,
    thinking_mode: str,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Follow the requested JSON output exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if thinking_mode in {"on", "off"}:
        payload["chat_template_kwargs"] = {
            "enable_thinking": thinking_mode == "on"
        }
    payload.update(extra_body)
    return payload


def write_one_rendered_prompt_per_class(
    *,
    output_dir: Path,
    specs: list[PromptSpec],
    sample: pd.DataFrame,
    transcript_cache: dict[str, str],
    modes: list[str],
    args: argparse.Namespace,
    extra_body: dict[str, Any],
) -> None:
    if sample.empty:
        return

    preview_dir = output_dir / "rendered_prompt_examples"
    preview_dir.mkdir(parents=True, exist_ok=True)

    first_row = sample.iloc[0].to_dict()
    seen: set[tuple[str, str]] = set()
    index_rows: list[dict[str, Any]] = []

    for spec in specs:
        template = spec.path.read_text(encoding="utf-8")
        transcript = transcript_cache[str(first_row["session_id"])]

        for mode in modes:
            class_key = (spec.family, mode)
            if class_key in seen:
                continue
            seen.add(class_key)

            rendered = render_prompt(
                template,
                first_row["learning_objective"],
                transcript,
            )
            base = safe_filename(f"{spec.family}__{mode}__{spec.id}")
            text_path = preview_dir / f"{base}.txt"
            json_path = preview_dir / f"{base}.request.json"

            text_path.write_text(rendered, encoding="utf-8")
            json_path.write_text(
                json.dumps(
                    build_request_payload(
                        args=args,
                        prompt=rendered,
                        thinking_mode=mode,
                        extra_body=extra_body,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            index_rows.append(
                {
                    "class": spec.family,
                    "thinking_mode": mode,
                    "prompt_id": spec.id,
                    "response_id": str(first_row["response_id"]),
                    "session_id": str(first_row["session_id"]),
                    "prompt_chars": len(rendered),
                    "text_file": text_path.name,
                    "request_file": json_path.name,
                }
            )
            print(
                f"[PROMPT EXAMPLE] class={spec.family} mode={mode} "
                f"prompt={spec.id} saved={json_path}",
                flush=True,
            )

    pd.DataFrame(index_rows).to_csv(
        preview_dir / "index.csv",
        index=False,
    )

def load_or_create_sample(
    *,
    args: argparse.Namespace,
    data: pd.DataFrame,
    selected_path: Path,
) -> pd.DataFrame:
    if args.resume and selected_path.exists():
        saved = pd.read_csv(
            selected_path,
            dtype={"response_id": str, "session_id": str},
        )
        if "response_id" not in saved.columns:
            raise ValueError(
                f"{selected_path} does not contain response_id"
            )

        saved_ids = saved["response_id"].astype(str).tolist()
        indexed = data.set_index("response_id", drop=False)
        missing_ids = [
            response_id
            for response_id in saved_ids
            if response_id not in indexed.index
        ]
        if missing_ids:
            raise ValueError(
                "Saved sample contains response IDs missing from current data: "
                f"{missing_ids[:10]}"
            )

        sample = indexed.loc[saved_ids].reset_index(drop=True)

        if "length_bin" in saved.columns:
            bin_map = dict(
                zip(
                    saved["response_id"].astype(str),
                    saved["length_bin"].astype(int),
                )
            )
            sample["length_bin"] = (
                sample["response_id"].map(bin_map).astype(int)
            )
        else:
            sample = assign_length_bins(sample, args.length_bins)

        print(
            f"[RESUME] Reusing {len(sample)} responses from {selected_path}. "
            "Sampling arguments are ignored.",
            flush=True,
        )
        return sample

    sample = balanced_sample(
        data,
        sample_size=args.sample_size,
        length_bins=args.length_bins,
        max_per_session=args.max_per_session,
        seed=args.seed,
    )
    sample.to_csv(selected_path, index=False)
    print(
        f"[SAMPLE] Saved {len(sample)} responses to {selected_path}",
        flush=True,
    )
    return sample


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_dir = Path(args.prompt_dir)
    transcripts_dir = Path(args.transcripts_dir)
    raw_path = output_dir / "raw_predictions.jsonl"
    selected_path = output_dir / "selected_samples.csv"

    extra_body = json.loads(args.extra_body_json)
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")

    features = pd.read_csv(
        args.features,
        dtype={"response_id": str, "session_id": str},
    )
    labels = pd.read_csv(
        args.labels,
        dtype={"response_id": str},
    )

    required_features = {
        "response_id",
        "session_id",
        "learning_objective",
    }
    missing = required_features - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    if "response_id" not in labels.columns:
        raise ValueError("Labels CSV must contain response_id")

    if "correct" in labels.columns:
        label_column = "correct"
    elif "is_correct" in labels.columns:
        label_column = "is_correct"
    else:
        raise ValueError(
            "Labels CSV must contain either correct or is_correct"
        )

    labels = labels[["response_id", label_column]].rename(
        columns={label_column: "correct"}
    )

    data = features.merge(
        labels,
        on="response_id",
        how="inner",
        validate="one_to_one",
    )
    data["correct"] = data["correct"].astype(float)
    data["learning_objective"] = (
        data["learning_objective"].map(sanitize_topic)
    )

    data, transcript_cache = add_transcript_metadata(
        data,
        transcripts_dir,
    )

    if raw_path.exists() and not args.resume:
        raw_path.unlink()
    if selected_path.exists() and not args.resume:
        selected_path.unlink()

    sample = load_or_create_sample(
        args=args,
        data=data,
        selected_path=selected_path,
    )

    specs = load_prompt_specs(prompt_dir, args.prompts)
    active_hashes = {
        spec.id: spec.prompt_hash
        for spec in specs
    }
    completed = completed_keys(raw_path) if args.resume else set()

    prevalence = float(data["correct"].mean())
    fallback = fallback_probability(
        args.null_fallback,
        prevalence,
    )

    sample_rows = sample.to_dict(orient="records")
    modes = thinking_modes(args.thinking_mode)

    write_one_rendered_prompt_per_class(
        output_dir=output_dir,
        specs=specs,
        sample=sample,
        transcript_cache=transcript_cache,
        modes=modes,
        args=args,
        extra_body=extra_body,
    )

    plan: list[
        tuple[PromptSpec, str, list[dict[str, Any]], int]
    ] = []
    total_new = 0

    for spec in specs:
        for mode in modes:
            pending: list[dict[str, Any]] = []
            skipped = 0

            for row in sample_rows:
                key = (
                    str(row["response_id"]),
                    spec.id,
                    mode,
                    spec.prompt_hash,
                )
                if key in completed:
                    skipped += 1
                else:
                    pending.append(row)

            plan.append((spec, mode, pending, skipped))
            total_new += len(pending)

    total_possible = len(sample) * len(specs) * len(modes)
    print(
        f"samples={len(sample)} prompts={len(specs)} modes={modes} "
        f"new_requests={total_new} "
        f"already_completed={total_possible - total_new}",
        flush=True,
    )

    global_number = 0

    for group_number, (spec, mode, pending, skipped) in enumerate(plan, 1):
        template = spec.path.read_text(encoding="utf-8")

        print(
            f"[PROMPT START {group_number}/{len(plan)}] "
            f"prompt={spec.id} mode={mode} "
            f"new={len(pending)} skipped={skipped}",
            flush=True,
        )

        if args.workers <= 1:
            for row in pending:
                record = run_request(
                    row=row,
                    spec=spec,
                    template=template,
                    transcript=transcript_cache[str(row["session_id"])],
                    args=args,
                    thinking_mode=mode,
                    extra_body=extra_body,
                )
                append_jsonl(raw_path, record)
                global_number += 1
                print_sample_result(
                    global_number,
                    total_new,
                    record,
                )
        else:
            with futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                future_map = {
                    executor.submit(
                        run_request,
                        row=row,
                        spec=spec,
                        template=template,
                        transcript=transcript_cache[
                            str(row["session_id"])
                        ],
                        args=args,
                        thinking_mode=mode,
                        extra_body=extra_body,
                    ): row
                    for row in pending
                }

                for future in futures.as_completed(future_map):
                    record = future.result()
                    append_jsonl(raw_path, record)
                    global_number += 1
                    print_sample_result(
                        global_number,
                        total_new,
                        record,
                    )

        print_prompt_summary(
            raw_path=raw_path,
            output_dir=output_dir,
            spec=spec,
            thinking_mode=mode,
            sample=sample,
            fallback=fallback,
            clip_epsilon=args.clip_epsilon,
            active_hashes=active_hashes,
            new_requests=len(pending),
            skipped_requests=skipped,
        )

    write_aggregates(
        raw_path=raw_path,
        output_dir=output_dir,
        fallback=fallback,
        clip_epsilon=args.clip_epsilon,
        active_hashes=active_hashes,
    )

    config = vars(args).copy()
    config.update(
        {
            "training_prevalence": prevalence,
            "resolved_null_fallback": fallback,
            "prompt_ids": [spec.id for spec in specs],
            "prompt_hashes": active_hashes,
            "prompt_schema_versions": {
                spec.id: spec.schema_version
                for spec in specs
            },
            "full_transcripts_preserved": True,
            "resume_reuses_saved_sample": True,
        }
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"finished: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
