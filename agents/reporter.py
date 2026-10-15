import html
import json
import re
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd
import plotly.express as px
from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.config import REPORTS_DIR
from core.llm import get_chat_model
from core.memory import store_document


def _validate_select(sql: str, registered_tables: set[str]) -> str | None:
    normalized = sql.strip().rstrip(";").strip()
    if not normalized.casefold().startswith("select ") or ";" in normalized:
        return "Only a single read-only SELECT query is allowed. Please revise the SQL."

    referenced_tables = re.findall(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not referenced_tables or any(table.casefold() not in registered_tables for table in referenced_tables):
        return "Queries may only read the registered Gold tables. Please revise the SQL."

    allowed_functions = {
        "abs", "avg", "cast", "coalesce", "count", "date_trunc", "max", "min",
        "nullif", "round", "sum", "try_cast",
    }
    function_calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", normalized)
    if any(function.casefold() not in allowed_functions for function in function_calls):
        return "The query uses a function that is not allowed. Please revise the SQL."
    return None


def run_reporter(
    gold_paths: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    scratchpad: dict[str, Any] = {"query_attempts": 0, "query_frame": None}
    connection = duckdb.connect(database=":memory:")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    @tool
    def inspect_gold_tables_tool() -> str:
        """Preview each Gold table's schema, row count, and first three rows."""
        previews = []
        for index, path in enumerate(gold_paths, start=1):
            frame = pd.read_parquet(path)
            previews.append(
                {
                    "table": f"gold_{index}",
                    "source_file": Path(path).name,
                    "row_count": len(frame),
                    "columns": [{"name": column, "type": str(dtype)} for column, dtype in frame.dtypes.items()],
                    "sample_rows": frame.head(3).to_dict(orient="records"),
                }
            )
        return json.dumps(previews, default=str)

    @tool
    def load_gold_data_tool() -> str:
        """Register the Gold Parquet files as DuckDB tables and return the table catalog."""
        catalog = []
        for index, path in enumerate(gold_paths, start=1):
            table_name = f"gold_{index}"
            frame = pd.read_parquet(path)
            connection.register(table_name, frame)
            catalog.append(
                {
                    "table": table_name,
                    "source_file": Path(path).name,
                    "row_count": len(frame),
                    "columns": [{"name": column, "type": str(dtype)} for column, dtype in frame.dtypes.items()],
                }
            )
        scratchpad["catalog"] = catalog
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql: str) -> str:
        """Execute a read-only SELECT query against loaded Gold tables; returns rows as JSON."""
        normalized = sql.strip().rstrip(";").strip()
        scratchpad["query_attempts"] += 1
        if scratchpad["query_attempts"] > 2:
            return "Query retry limit reached; only two query attempts are allowed per report."
        validation_error = _validate_select(
            normalized,
            {item["table"].casefold() for item in scratchpad.get("catalog", [])},
        )
        if validation_error:
            return validation_error
        try:
            frame = connection.execute(normalized).fetchdf()
            scratchpad["query_frame"] = frame
            scratchpad["sql"] = normalized
            return json.dumps(frame.to_dict(orient="records"), default=str)
        except Exception as error:
            return f"SQL error on attempt {scratchpad['query_attempts']}/2: {error}"

    @tool
    def build_report_tool(
        summary: str,
        chart_type: Literal["bar", "line", "pie", "scatter"] = "bar",
    ) -> str:
        """Create and save the HTML report with the answer, query results, and a chart."""
        frame = scratchpad.get("query_frame")
        chart_html = ""
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            numeric_columns = list(frame.select_dtypes(include="number").columns)
            category_columns = [column for column in frame.columns if column not in numeric_columns]
            if numeric_columns:
                x_column = category_columns[0] if category_columns else numeric_columns[0]
                y_column = numeric_columns[0] if category_columns else (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
                try:
                    if chart_type == "line":
                        figure = px.line(frame, x=x_column, y=y_column)
                    elif chart_type == "pie":
                        figure = px.pie(frame, names=x_column, values=y_column)
                    elif chart_type == "scatter":
                        figure = px.scatter(frame, x=x_column, y=y_column)
                    else:
                        figure = px.bar(frame, x=x_column, y=y_column)
                    chart_html = figure.to_html(full_html=False, include_plotlyjs="cdn")
                except Exception:
                    chart_html = ""

        table_html = (
            frame.to_html(index=False, escape=True, classes="results")
            if isinstance(frame, pd.DataFrame) and not frame.empty
            else "<p>The query returned no rows.</p>"
        )
        escaped_summary = html.escape(summary)
        sql = html.escape(scratchpad.get("sql", "No SQL query was executed."))
        report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline Report {html.escape(run_id[:8])}</title>
<style>
:root {{ color-scheme: light; font-family: Segoe UI, sans-serif; color: #17211d; background: #f4f6f4; }}
body {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px; }}
header {{ border-bottom: 3px solid #167c63; padding-bottom: 16px; margin-bottom: 24px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin-top: 30px; font-size: 18px; }}
.summary {{ background: #e6f2ed; border-left: 4px solid #167c63; padding: 16px; white-space: pre-wrap; }}
.results {{ border-collapse: collapse; width: 100%; background: white; }}
.results th, .results td {{ border: 1px solid #d6ded9; padding: 9px 12px; text-align: left; }}
.results th {{ background: #eaf0ed; }}
code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<header><h1>Retail Data Analysis</h1><div>Run {html.escape(run_id)}</div></header>
<h2>Business question</h2><p>{html.escape(business_intent)}</p>
<h2>Analysis</h2><div class="summary">{escaped_summary}</div>
<h2>Results</h2>{table_html}
<h2>Query</h2><pre><code>{sql}</code></pre>
<h2>Chart</h2>{chart_html or '<p>No numeric results were available to chart.</p>'}
</body>
</html>"""
        output_path = REPORTS_DIR / f"report_{run_id}.html"
        output_path.write_text(report_html, encoding="utf-8")
        store_document(
            f"report-{run_id}",
            f"Question: {business_intent}\nSummary: {summary}\nReport: {output_path}",
            {"kind": "report", "run_id": run_id},
        )
        scratchpad["report_path"] = str(output_path)
        return str(output_path)

    tools = [inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, build_report_tool]
    try:
        result = invoke_agent(
            agent_name="reporter",
            model=get_chat_model(),
            tools=tools,
            system_prompt=(
                "You are a precise retail data analyst. Inspect the Gold tables, load them into "
                "DuckDB, and write one read-only SELECT query that answers the user's question. "
                "If SQL fails, use the returned error to revise and retry once. Then call "
                "build_report_tool with a concise, evidence-based summary and the best chart type. "
                "Never invent results."
            ),
            task=task_description,
            run_id=run_id,
            inputs={"gold_paths": gold_paths, "business_intent": business_intent},
        )
        if "report_path" not in scratchpad:
            messages = result.get("messages", [])
            final_text = str(getattr(messages[-1], "content", "Analysis completed.")) if messages else "Analysis completed."
            build_report_tool.invoke({"summary": final_text, "chart_type": "bar"})
        return scratchpad["report_path"]
    finally:
        connection.close()
