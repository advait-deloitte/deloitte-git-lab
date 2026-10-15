import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.config import PROFILES_DIR
from core.llm import get_chat_model


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _semantic_hint(column_name: str) -> str:
    name = column_name.casefold()
    hints = {
        "date": "date or time",
        "time": "date or time",
        "product": "product identifier or description",
        "sku": "product identifier",
        "customer": "customer identifier or attribute",
        "region": "geographic region",
        "country": "geographic location",
        "price": "unit price or monetary value",
        "amount": "monetary value",
        "revenue": "sales revenue",
        "sales": "sales measure",
        "quantity": "units sold",
        "order": "order identifier or attribute",
        "id": "identifier or key",
    }
    for token, hint in hints.items():
        if token in name:
            return hint
    return "meaning inferred from the column name and sample values"


def make_profiler_tools(
    file_paths: list[str],
    business_intent: str,
    run_id: str,
    scratchpad: dict[str, Any],
) -> list[Any]:
    @tool
    def inspect_files_tool() -> str:
        """Preview each source CSV's shape, columns, inferred types, and sample values."""
        previews = []
        for file_path in file_paths:
            frame = pd.read_csv(file_path, nrows=5)
            previews.append(
                {
                    "file": Path(file_path).name,
                    "shape_preview_rows": len(frame),
                    "columns": list(frame.columns),
                    "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                    "samples": frame.head(3).where(pd.notna(frame), None).to_dict(orient="records"),
                }
            )
        scratchpad["previews"] = previews
        return json.dumps(previews, default=str)

    @tool
    def profiler_tool() -> str:
        """Calculate full-file column statistics and save a combined JSON profile."""
        files = []
        quality_notes = []
        semantic_meanings = {}
        columns_by_file = []

        for file_path in file_paths:
            frame = pd.read_csv(file_path)
            column_profiles = []
            for column in frame.columns:
                series = frame[column]
                null_count = int(series.isna().sum())
                profile: dict[str, Any] = {
                    "name": str(column),
                    "dtype": str(series.dtype),
                    "null_count": null_count,
                    "null_percent": round(null_count / len(frame) * 100, 2) if len(frame) else 0.0,
                    "unique_count": int(series.nunique(dropna=True)),
                    "sample_values": [_json_value(value) for value in series.dropna().head(5).tolist()],
                    "semantic_meaning": _semantic_hint(str(column)),
                }
                if pd.api.types.is_numeric_dtype(series):
                    profile.update(
                        {
                            "min": _json_value(series.min()),
                            "max": _json_value(series.max()),
                            "mean": _json_value(series.mean()),
                        }
                    )
                if profile["null_percent"] >= 20:
                    quality_notes.append(
                        f"{Path(file_path).name}.{column} has {profile['null_percent']}% null values."
                    )
                if len(frame) and profile["unique_count"] / len(frame) > 0.9:
                    quality_notes.append(
                        f"{Path(file_path).name}.{column} has high cardinality and may be an identifier."
                    )
                column_profiles.append(profile)

            columns_by_file.append({str(column).casefold() for column in frame.columns})
            semantic_meanings[Path(file_path).name] = {
                item["name"]: item["semantic_meaning"] for item in column_profiles
            }
            files.append(
                {
                    "path": str(Path(file_path).resolve()),
                    "name": Path(file_path).name,
                    "row_count": len(frame),
                    "column_count": len(frame.columns),
                    "duplicate_rows": int(frame.duplicated().sum()),
                    "columns": column_profiles,
                }
            )

        join_keys = []
        if len(columns_by_file) > 1:
            shared_columns = set.intersection(*columns_by_file)
            join_keys = sorted(
                column
                for column in shared_columns
                if column.endswith("_id") or column == "id"
            )

        profile = {
            "run_id": run_id,
            "business_intent": business_intent,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "semantic_meanings": semantic_meanings,
            "join_keys": join_keys,
            "quality_notes": quality_notes,
        }
        output_path = PROFILES_DIR / f"profile_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id[:8]}.json"
        output_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        scratchpad["profile_path"] = str(output_path)
        scratchpad["profile"] = profile
        return str(output_path)

    return [inspect_files_tool, profiler_tool]


def run_profiler(file_paths: list[str], business_intent: str, run_id: str) -> str:
    scratchpad: dict[str, Any] = {}
    tools = make_profiler_tools(file_paths, business_intent, run_id, scratchpad)
    task = (
        f"Profile these CSV files for the business question: {business_intent}. "
        "First call inspect_files_tool to review their structure, then call profiler_tool "
        "to calculate full statistics and save the combined profile."
    )
    result = invoke_agent(
        agent_name="profiler",
        model=get_chat_model(),
        tools=tools,
        system_prompt=(
            "You are a careful data profiling agent. Inspect the files before calculating "
            "statistics. Use the provided tools and do not invent data."
        ),
        task=task,
        run_id=run_id,
        inputs={"file_paths": file_paths, "business_intent": business_intent},
    )
    if "profile_path" not in scratchpad:
        tools[1].invoke({})
    return scratchpad["profile_path"]
