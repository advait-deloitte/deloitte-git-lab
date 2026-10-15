import json
import uuid
from datetime import datetime, timezone
from typing import Any

from core.config import AUDIT_DIR


class AuditLogger:
    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = str(run_id) if run_id is not None else str(uuid.uuid4())
        self.log_file = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent: str, action: str, **extra: Any) -> None:
        entry = {
            **extra,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
        }
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, default=str) + "\n")

    def get_logs(self) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []

        with self.log_file.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
