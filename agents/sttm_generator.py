import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.config import STTM_DIR
from core.llm import get_chat_model
from core.state import PipelineState

STTM_COLUMNS = [
    "layer",
    "source_file",
    "source_table",
    "source_col",
    "target_table",
    "target_col",
    "data_type",
    "transformation_logic",
    "null_handling",
    "aggregation",
    "group_by",
    "join_key",
    "join_type",
]


def _read_sources(layer: str, state: PipelineState) -> list[tuple[str, pd.DataFrame]]:
    if layer == "bronze":
        return []
    paths = state.bronze_output_paths if layer == "silver" else state.silver_output_paths
    return [(path, pd.read_parquet(path)) for path in paths]


def _bronze_rules(state: PipelineState) -> list[dict[str, Any]]:
    profile = json.loads(Path(state.profile_path).read_text(encoding="utf-8"))
    rules = []
    for file_profile in profile["files"]:
        for column in file_profile["columns"]:
            rules.append(
                {
                    "layer": "bronze",
                    "source_file": file_profile["path"],
                    "source_table": Path(file_profile["name"]).stem,
                    "source_col": column["name"],
                    "target_table": Path(file_profile["name"]).stem,
                    "target_col": column["name"],
                    "data_type": column["dtype"],
                    "transformation_logic": "preserve_raw_value",
                    "null_handling": "preserve",
                    "aggregation": "",
                    "group_by": False,
                    "join_key": False,
                    "join_type": "",
                }
            )
    return rules


def _silver_rules(state: PipelineState) -> list[dict[str, Any]]:
    rules = []
    for path, frame in _read_sources("silver", state):
        table_name = Path(path).name.split("__bronze__", 1)[0]
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                logic, null_handling, data_type = "parse_datetime", "preserve", "datetime"
            elif pd.api.types.is_integer_dtype(series):
                logic, null_handling, data_type = "cast_integer", "fill_median", "integer"
            elif pd.api.types.is_numeric_dtype(series):
                logic, null_handling, data_type = "cast_numeric", "fill_mean", "numeric"
            else:
                logic, null_handling, data_type = "strip_text", "fill_mode", "string"
                if any(token in str(column).casefold() for token in ("date", "time")):
                    logic, null_handling, data_type = "parse_datetime", "preserve", "datetime"
            rules.append(
                {
                    "layer": "silver",
                    "source_file": str(Path(path).resolve()),
                    "source_table": table_name,
                    "source_col": str(column),
                    "target_table": table_name,
                    "target_col": str(column).strip().lower().replace(" ", "_"),
                    "data_type": data_type,
                    "transformation_logic": logic,
                    "null_handling": null_handling,
                    "aggregation": "",
                    "group_by": False,
                    "join_key": False,
                    "join_type": "",
                }
            )
    return rules


def _gold_rules(state: PipelineState) -> list[dict[str, Any]]:
    sources = [
        (Path(path).name.split("__silver__", 1)[0], path, frame)
        for path, frame in _read_sources("gold", state)
    ]
    if not sources:
        raise ValueError("Gold mapping requires at least one Silver input.")

    join_keys: set[str] = set()
    if len(sources) > 1:
        common_columns = set.intersection(*(set(map(str, frame.columns)) for _, _, frame in sources))
        join_keys = {
            column
            for column in common_columns
            if column.casefold() == "id" or column.casefold().endswith("_id")
            if not column.casefold().startswith("pk_")
        }
        if state.profile_path and Path(state.profile_path).exists():
            profile = json.loads(Path(state.profile_path).read_text(encoding="utf-8"))
            profiled_keys = set(profile.get("join_keys", []))
            matched_profile_keys = join_keys & profiled_keys
            if matched_profile_keys:
                join_keys = matched_profile_keys
        if not join_keys:
            raise ValueError(
                "Silver inputs have no shared *_id column for a safe Gold outer join. "
                "Review the Silver STTM or provide matching business keys."
            )

    intent_tokens = set(state.business_intent.casefold().replace("?", " ").split())
    column_counts: dict[str, int] = {}
    for _, _, frame in sources:
        for column in frame.columns:
            column_counts[str(column)] = column_counts.get(str(column), 0) + 1

    target_table = "gold_analysis"
    intent = state.business_intent.casefold()
    aggregation = "avg" if any(word in intent for word in ("average", "avg", "mean")) else "sum"
    rules: list[dict[str, Any]] = []

    for table_name, path, frame in sources:
        for key in sorted(join_keys):
            rules.append(
                {
                    "layer": "gold",
                    "source_file": str(Path(path).resolve()),
                    "source_table": table_name,
                    "source_col": key,
                    "target_table": target_table,
                    "target_col": key,
                    "data_type": str(frame[key].dtype),
                    "transformation_logic": "outer_join",
                    "null_handling": "preserve",
                    "aggregation": "",
                    "group_by": False,
                    "join_key": True,
                    "join_type": "outer",
                }
            )

        categorical = [
            column
            for column in frame.columns
            if str(column) not in join_keys
            and not pd.api.types.is_numeric_dtype(frame[column])
            and not any(token in str(column).casefold() for token in ("pk_", "_id", "date", "time"))
            and (len(frame) < 20 or frame[column].nunique(dropna=True) / max(len(frame), 1) < 0.8)
        ]
        numeric = [
            column
            for column in frame.select_dtypes(include="number").columns
            if str(column) not in join_keys
            and not any(token in str(column).casefold() for token in ("id", "zip", "postal", "year"))
        ]
        categorical.sort(
            key=lambda column: sum(token in str(column).casefold() for token in intent_tokens),
            reverse=True,
        )
        numeric.sort(
            key=lambda column: sum(token in str(column).casefold() for token in intent_tokens)
            + 3 * any(token in str(column).casefold() for token in ("sales", "revenue", "amount", "price", "quantity", "profit")),
            reverse=True,
        )

        for column in categorical[:2]:
            name = str(column)
            target_column = f"{table_name}_{name}" if column_counts[name] > 1 else name
            rules.append(
                {
                    "layer": "gold",
                    "source_file": str(Path(path).resolve()),
                    "source_table": table_name,
                    "source_col": name,
                    "target_table": target_table,
                    "target_col": target_column,
                    "data_type": str(frame[column].dtype),
                    "transformation_logic": "group_by",
                    "null_handling": "drop_null_group",
                    "aggregation": "",
                    "group_by": True,
                    "join_key": False,
                    "join_type": "",
                }
            )
        for column in numeric:
            name = str(column)
            rules.append(
                {
                    "layer": "gold",
                    "source_file": str(Path(path).resolve()),
                    "source_table": table_name,
                    "source_col": name,
                    "target_table": target_table,
                    "target_col": f"{table_name}_{name}_{aggregation}",
                    "data_type": str(frame[column].dtype),
                    "transformation_logic": aggregation,
                    "null_handling": "ignore_null",
                    "aggregation": aggregation,
                    "group_by": False,
                    "join_key": False,
                    "join_type": "",
                }
            )
        if not numeric:
            rules.append(
                {
                    "layer": "gold",
                    "source_file": str(Path(path).resolve()),
                    "source_table": table_name,
                    "source_col": "",
                    "target_table": target_table,
                    "target_col": f"{table_name}_record_count",
                    "data_type": "integer",
                    "transformation_logic": "count",
                    "null_handling": "",
                    "aggregation": "count",
                    "group_by": False,
                    "join_key": False,
                    "join_type": "",
                }
            )
    return rules


def run_sttm_agent(state: PipelineState, layer: str) -> str:
    if layer not in {"bronze", "silver", "gold"}:
        raise ValueError(f"Unsupported STTM layer: {layer}")
    scratchpad: dict[str, Any] = {}

    @tool
    def inspect_context_tool() -> str:
        """Preview the source schema and relevant business context for this mapping."""
        if layer == "bronze":
            context = json.loads(Path(state.profile_path).read_text(encoding="utf-8"))
        else:
            sources = _read_sources(layer, state)
            context = {
                "business_intent": state.business_intent,
                "sources": [
                    {"file": Path(path).name, "columns": list(frame.columns), "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()}}
                    for path, frame in sources
                ],
            }
        return json.dumps(context, default=str)[:12000]

    def save_rules(requested_layer: str) -> str:
        if requested_layer != layer:
            raise ValueError(f"This run is generating {layer} rules, not {requested_layer} rules.")
        rules = {
            "bronze": _bronze_rules,
            "silver": _silver_rules,
            "gold": _gold_rules,
        }[layer](state)
        output_path = STTM_DIR / f"sttm_{layer}_{state.run_id}.csv"
        pd.DataFrame(rules, columns=STTM_COLUMNS).to_csv(output_path, index=False)
        scratchpad["sttm_path"] = str(output_path)
        return f"Saved {len(rules)} {layer} mapping rows to {output_path}."

    @tool
    def generate_bronze_sttm_tool() -> str:
        """Generate direct source-to-target mappings for the Bronze layer."""
        return save_rules("bronze")

    @tool
    def generate_silver_sttm_tool() -> str:
        """Generate cleansing, type-casting, null-handling, and deduplication mappings for Silver."""
        return save_rules("silver")

    @tool
    def generate_gold_sttm_tool() -> str:
        """Generate Gold grouping and aggregation mappings from Silver schemas and user intent."""
        return save_rules("gold")

    tools = [
        inspect_context_tool,
        generate_bronze_sttm_tool,
        generate_silver_sttm_tool,
        generate_gold_sttm_tool,
    ]
    task = (
        f"Create the {layer.title()} source-to-target mapping for this pipeline. "
        f"Business intent: {state.business_intent}. First inspect the context, then call only "
        f"generate_{layer}_sttm_tool. The output must be saved for human review."
    )
    invoke_agent(
        agent_name=f"sttm_{layer}",
        model=get_chat_model(),
        tools=tools,
        system_prompt=(
            "You are a data mapping agent. Inspect the available schema and business intent, "
            "then use the matching layer's generation tool. Never claim a mapping was saved "
            "unless the tool confirms it."
        ),
        task=task,
        run_id=state.run_id,
        inputs={"layer": layer, "business_intent": state.business_intent},
    )
    if "sttm_path" not in scratchpad:
        save_rules(layer)
    return scratchpad["sttm_path"]
