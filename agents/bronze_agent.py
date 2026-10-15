from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.config import BRONZE_DIR
from core.llm import get_chat_model


def ingest_bronze(file_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    mappings = pd.read_csv(sttm_path)
    outputs = []
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in file_paths:
        source = Path(file_path).resolve()
        frame = pd.read_csv(source)
        rules = mappings
        if "source_file" in rules.columns:
            matching = rules[
                rules["source_file"].fillna("").map(
                    lambda value: str(Path(value).resolve()).casefold() if value else ""
                )
                == str(source).casefold()
            ]
            if not matching.empty:
                rules = matching
        if "source_table" in rules.columns and rules is mappings:
            matching = rules[rules["source_table"].astype(str) == source.stem]
            if not matching.empty:
                rules = matching

        column_map = {
            str(row.source_col): str(row.target_col)
            for row in rules.itertuples()
            if pd.notna(row.source_col)
            and str(row.source_col) in frame.columns
            and pd.notna(row.target_col)
            and str(row.target_col)
        }
        if not column_map:
            raise ValueError(f"No approved Bronze mappings matched {source.name}.")
        output = frame[list(column_map)].rename(columns=column_map)
        output_path = BRONZE_DIR / f"{source.stem}__bronze__{run_id[:8]}.parquet"
        output.to_parquet(output_path, index=False)
        outputs.append(str(output_path))

    return outputs


def run_bronze_agent(
    file_paths: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    scratchpad: dict[str, Any] = {}

    @tool
    def inspect_task_tool() -> str:
        """Preview source CSV data and the approved Bronze transformation rules."""
        previews = []
        for file_path in file_paths:
            frame = pd.read_csv(file_path, nrows=5)
            previews.append({"file": Path(file_path).name, "columns": list(frame.columns), "sample": frame.to_dict(orient="records")})
        rules = pd.read_csv(sttm_path).fillna("").to_dict(orient="records")
        return str({"previews": previews, "approved_rules": rules})[:12000]

    @tool
    def bronze_ingestion_tool() -> str:
        """Execute the approved source mappings and write one Bronze Parquet file per CSV."""
        scratchpad["paths"] = ingest_bronze(file_paths, sttm_path, run_id)
        return str(scratchpad["paths"])

    tools = [inspect_task_tool, bronze_ingestion_tool]
    invoke_agent(
        agent_name="bronze",
        model=get_chat_model(),
        tools=tools,
        system_prompt=(
            "You are a Bronze ingestion agent. Inspect the input CSVs and approved mapping "
            "rules before calling bronze_ingestion_tool. Preserve raw values; do not invent "
            "or alter transformation rules."
        ),
        task=task_description,
        run_id=run_id,
        inputs={"file_paths": file_paths, "sttm_path": sttm_path},
    )
    if "paths" not in scratchpad:
        tools[1].invoke({})
    return scratchpad["paths"]
