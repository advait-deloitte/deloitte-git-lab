import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.config import SILVER_DIR
from core.llm import get_chat_model


def _source_table(path: str) -> str:
    return Path(path).name.split("__bronze__", 1)[0]


def _clean_series(series: pd.Series, rule: Any) -> pd.Series:
    logic = str(rule.transformation_logic).casefold()
    null_handling = str(rule.null_handling).casefold()

    if logic == "parse_datetime":
        cleaned = pd.to_datetime(series, errors="coerce")
    elif logic in {"cast_integer", "cast_numeric"}:
        cleaned = pd.to_numeric(series, errors="coerce")
        if logic == "cast_integer":
            cleaned = cleaned.round().astype("Int64")
    elif logic == "strip_text":
        cleaned = series.astype("string").str.strip().replace("", pd.NA)
    else:
        cleaned = series.copy()

    if cleaned.notna().any():
        if null_handling == "fill_mean" and pd.api.types.is_numeric_dtype(cleaned):
            cleaned = cleaned.fillna(cleaned.mean())
        elif null_handling == "fill_median" and pd.api.types.is_numeric_dtype(cleaned):
            cleaned = cleaned.fillna(cleaned.median())
        elif null_handling == "fill_mode":
            modes = cleaned.mode(dropna=True)
            if not modes.empty:
                cleaned = cleaned.fillna(modes.iloc[0])
    return cleaned


def cleanse_silver(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    mappings = pd.read_csv(sttm_path).fillna("")
    outputs = []
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    for bronze_path in bronze_paths:
        table = _source_table(bronze_path)
        source = pd.read_parquet(bronze_path)
        rules = mappings[mappings["source_table"].astype(str) == table]
        if rules.empty:
            raise ValueError(f"No approved Silver mappings found for {table}.")

        selected: dict[str, pd.Series] = {}
        for rule in rules.itertuples(index=False):
            source_column = str(rule.source_col)
            target_column = str(rule.target_col)
            if source_column not in source.columns or not target_column:
                continue
            selected[target_column] = _clean_series(source[source_column], rule)
        if not selected:
            raise ValueError(f"No approved Silver columns were present in {bronze_path}.")

        output = pd.DataFrame(selected).drop_duplicates().reset_index(drop=True)
        identifier = re.sub(r"[^a-zA-Z0-9_]+", "_", table).strip("_").lower() or "table"
        output.insert(0, f"pk_{identifier}_silver_id", range(1, len(output) + 1))
        output_path = SILVER_DIR / f"{table}__silver__{run_id[:8]}.parquet"
        output.to_parquet(output_path, index=False)
        outputs.append(str(output_path))

    return outputs


def run_silver_agent(
    bronze_paths: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    scratchpad: dict[str, Any] = {}

    @tool
    def inspect_task_tool() -> str:
        """Preview Bronze schemas and the approved Silver cleansing rules."""
        previews = []
        for path in bronze_paths:
            frame = pd.read_parquet(path)
            previews.append(
                {
                    "file": Path(path).name,
                    "columns": list(frame.columns),
                    "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                    "null_counts": frame.isna().sum().to_dict(),
                    "sample": frame.head(3).to_dict(orient="records"),
                }
            )
        rules = pd.read_csv(sttm_path).fillna("").to_dict(orient="records")
        return str({"previews": previews, "approved_rules": rules})[:12000]

    @tool
    def silver_ingestion_tool() -> str:
        """Apply approved null, deduplication, type-casting, and text rules; write Silver Parquet."""
        scratchpad["paths"] = cleanse_silver(bronze_paths, sttm_path, run_id)
        return str(scratchpad["paths"])

    tools = [inspect_task_tool, silver_ingestion_tool]
    invoke_agent(
        agent_name="silver",
        model=get_chat_model(),
        tools=tools,
        system_prompt=(
            "You are a Silver cleansing agent. Inspect the Bronze schemas and approved rules, "
            "then call silver_ingestion_tool. Keep only approved target columns and do not "
            "invent transformations."
        ),
        task=task_description,
        run_id=run_id,
        inputs={"bronze_paths": bronze_paths, "sttm_path": sttm_path},
    )
    if "paths" not in scratchpad:
        tools[1].invoke({})
    return scratchpad["paths"]
