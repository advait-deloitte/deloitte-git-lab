from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    def __init__(self, agent_name: str, run_id: str) -> None:
        self.agent_name = agent_name
        self.run_id = str(run_id)
        self.start_time = time.time()
        self.trace: dict[str, Any] = {
            "agent": agent_name,
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": None,
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": None,
            "status": "running",
        }

    def set_input(self, **kwargs: Any) -> AgentTrace:
        self.trace["input"] = kwargs
        return self

    def set_plan(self, plan_str: str) -> AgentTrace:
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs: Any) -> AgentTrace:
        self.trace["output"] = kwargs
        return self

    def extract_from_messages(self, messages: Iterable[Any]) -> AgentTrace:
        for message in messages:
            if isinstance(message, HumanMessage):
                self.trace["reasoning_steps"].append(
                    {"type": "task_input", "content": message.content}
                )
            elif isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    tool_calls = getattr(message, "additional_kwargs", {}).get(
                        "tool_calls", []
                    )
                self.trace["tool_calls"].extend(tool_calls)

                content = message.content
                if content and (not isinstance(content, str) or content.strip()):
                    self.trace["reasoning_steps"].append(
                        {"type": "ai_reasoning", "content": content}
                    )
                    if not self.trace["plan"]:
                        self.trace["plan"] = content
            elif isinstance(message, ToolMessage):
                tool_result = {"type": "tool_result", "content": message.content}
                if message.name:
                    tool_result["tool_name"] = message.name
                if message.tool_call_id:
                    tool_result["tool_call_id"] = message.tool_call_id
                self.trace["reasoning_steps"].append(tool_result)

        return self

    def complete(self, status: str = "success") -> AgentTrace:
        self.trace["duration_seconds"] = time.time() - self.start_time
        self.trace["status"] = status

        trace_path = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as file:
            json.dump(self.trace, file, indent=2, default=str)

        print(
            f"Trace {status}: agent={self.agent_name} run_id={self.run_id} "
            f"duration={self.trace['duration_seconds']:.3f}s path={trace_path}"
        )
        return self

    def fail(self, error: Any) -> AgentTrace:
        self.trace["error"] = str(error)
        return self.complete(status="failed")
