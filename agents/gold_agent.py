from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.config import GOLD_DIR
from core.llm import get_chat_model

_AGGREGATIONS = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "max": "max",
    "min": "min",
}


def _table_name(path: str) -> str:
    return Path(path).name.split("__silver__", 1)[0]


def materialize_gold(silver_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    mappings = pd.read_csv(sttm_path).fillna("")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for target_table, target_rules in mappings.groupby("target_table", sort=False):
        mapped_frames = []
        join_columns: set[str] = set()
        count_markers: dict[str, str] = {}

        for silver_path in silver_paths:
            source_table = _table_name(silver_path)
            table_rules = target_rules[target_rules["source_table"].astype(str) == source_table]
            if table_rules.empty:
                continue

            source_frame = pd.read_parquet(silver_path)
            column_map: dict[str, str] = {}
            for rule in table_rules.itertuples(index=False):
                source_column = str(rule.source_col)
                target_column = str(rule.target_col)
                if source_column and source_column in source_frame.columns and target_column:
                    column_map[source_column] = target_column
                if (
                    "join_key" in table_rules.columns
                    and str(rule.join_key).casefold() in {"true", "1", "yes"}
                    and target_column
                ):
                    join_columns.add(target_column)

            if not column_map:
                raise ValueError(f"No source columns from {source_table} match its approved Gold rules.")
            mapped = source_frame[list(column_map)].rename(columns=column_map)
            for rule in table_rules.itertuples(index=False):
                if str(rule.aggregation).casefold() == "count" and not str(rule.source_col):
                    target_column = str(rule.target_col) or f"{source_table}_record_count"
                    marker_column = f"__count_marker_{target_column}"
                    mapped[marker_column] = 1
                    count_markers[target_column] = marker_column
            mapped_frames.append((source_table, mapped))

        if not mapped_frames:
            raise ValueError(f"No Silver inputs match the Gold target table {target_table}.")

        frame = mapped_frames[0][1]
        for source_table, next_frame in mapped_frames[1:]:
            keys = [
                key
                for key in sorted(join_columns)
                if key in frame.columns and key in next_frame.columns
            ]
            if not keys:
                raise ValueError(
                    f"Cannot outer-join {source_table} to {target_table}: "
                    "no approved join key exists in both mapped inputs."
                )
            frame = frame.merge(next_frame, how="outer", on=keys, validate="many_to_many")

        selected_dimensions = [
            str(rule.target_col)
            for rule in target_rules.itertuples(index=False)
            if str(rule.group_by).casefold() in {"true", "1", "yes"}
            and str(rule.target_col) in frame.columns
        ]
        selected_dimensions = list(dict.fromkeys(selected_dimensions))
        aggregations: dict[str, tuple[str, str]] = {}
        count_targets = []
        for rule in target_rules.itertuples(index=False):
            source_column = str(rule.source_col)
            target_column = str(rule.target_col)
            aggregation = str(rule.aggregation).casefold()
            if aggregation == "count" and not source_column:
                count_targets.append(target_column or "record_count")
            elif aggregation in _AGGREGATIONS and source_column and target_column in frame.columns:
                aggregations[target_column] = (target_column, _AGGREGATIONS[aggregation])

        named_aggregations = {
            target: pd.NamedAgg(column=source, aggfunc=aggregation)
            for target, (source, aggregation) in aggregations.items()
        }
        if selected_dimensions and named_aggregations:
            output = frame.groupby(selected_dimensions, dropna=False).agg(**named_aggregations).reset_index()
        elif named_aggregations:
            output = pd.DataFrame(
                [{target: getattr(frame[source], aggregation)() for target, (source, aggregation) in aggregations.items()}]
            )
        else:
            output = frame[selected_dimensions].drop_duplicates().reset_index(drop=True) if selected_dimensions else pd.DataFrame(index=[0])

        for target in count_targets:
            marker_column = count_markers.get(target)
            if marker_column is None or marker_column not in frame.columns:
                continue
            if selected_dimensions:
                counts = (
                    frame.groupby(selected_dimensions, dropna=False)[marker_column]
                    .sum()
                    .rename(target)
                    .reset_index()
                )
                output = output.merge(counts, on=selected_dimensions, how="outer")
            else:
                output[target] = [frame[marker_column].fillna(0).sum()]

        if not selected_dimensions and not aggregations and not count_targets:
            raise ValueError(f"Gold mappings for {target_table} contain no executable aggregations.")

        output.insert(0, "pk_gold_id", range(1, len(output) + 1))
        safe_name = "".join(character if character.isalnum() or character in "_-" else "_" for character in str(target_table))
        output_path = GOLD_DIR / f"{safe_name}__{run_id[:8]}.parquet"
        output.to_parquet(output_path, index=False)
        outputs.append(str(output_path))

    return outputs


def run_gold_agent(
    silver_paths: list[str],
    sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    scratchpad: dict[str, Any] = {}

    @tool
    def inspect_task_tool() -> str:
        """Preview Silver schemas and the approved Gold grouping and aggregation rules."""
        previews = []
        for path in silver_paths:
            frame = pd.read_parquet(path)
            previews.append(
                {
                    "file": Path(path).name,
                    "columns": list(frame.columns),
                    "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                    "sample": frame.head(3).to_dict(orient="records"),
                }
            )
        rules = pd.read_csv(sttm_path).fillna("").to_dict(orient="records")
        return str({"business_intent": business_intent, "previews": previews, "approved_rules": rules})[:12000]

    @tool
    def gold_ingestion_tool() -> str:
        """Materialize approved Gold tables by grouping Silver data and applying aggregates."""
        scratchpad["paths"] = materialize_gold(silver_paths, sttm_path, run_id)
        return str(scratchpad["paths"])

    tools = [inspect_task_tool, gold_ingestion_tool]
    invoke_agent(
        agent_name="gold",
        model=get_chat_model(),
        tools=tools,
        system_prompt=(
            "You are a Gold analytics agent. Inspect the Silver schemas and approved Gold rules, "
            "then call gold_ingestion_tool. Apply only the approved grouping and aggregation rules."
        ),
        task=task_description,
        run_id=run_id,
        inputs={"silver_paths": silver_paths, "sttm_path": sttm_path, "business_intent": business_intent},
    )
    if "paths" not in scratchpad:
        tools[1].invoke({})
    return scratchpad["paths"]
