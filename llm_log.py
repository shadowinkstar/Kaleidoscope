from pathlib import Path
from datetime import datetime
import json

DEFAULT_LOG_PATH = Path("logs/llm_io.json")


def record_llm_io(function: str, llm_input, llm_output, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Record LLM input and output to a JSON file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "function": function,
        "input": llm_input,
        "output": llm_output,
    }
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    log_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_llm_log(log_path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    """Load recorded LLM IO data from JSON file."""
    if log_path.exists():
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []
