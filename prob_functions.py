import math
import json
import re

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
 