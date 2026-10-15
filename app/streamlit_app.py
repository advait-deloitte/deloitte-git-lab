import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agents.orchestrator import APPROVAL_STATUSES, advance_pipeline, create_pipeline_state
from core.config import LANDING_DIR, LLM_PROVIDER
from core.state import PipelineState

st.set_page_config(page_title="Medallion Pipeline", page_icon="📊", layout="wide")


def _provider_has_key() -> bool:
    from core import config

    key_by_provider = {
        "anthropic": config.ANTHROPIC_API_KEY,
        "github": config.GITHUB_TOKEN,
        "openai": config.OPENAI_API_KEY,
        "groq": config.GROQ_API_KEY,
        "gemini": config.GOOGLE_API_KEY,
    }
    return bool(key_by_provider.get(LLM_PROVIDER))


def _save_uploads(uploads: list[Any], run_id: str) -> list[str]:
    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for upload in uploads:
        safe_name = Path(upload.name).name
        destination = LANDING_DIR / f"{run_id[:8]}_{safe_name}"
        destination.write_bytes(upload.getvalue())
        paths.append(str(destination.resolve()))
    return paths


def _approval_path(state: PipelineState) -> str:
    return {
        "awaiting_bronze_approval": state.sttm_bronze_path,
        "awaiting_silver_approval": state.sttm_silver_path,
        "awaiting_gold_approval": state.sttm_gold_path,
    }[state.status]


def _approval_label(state: PipelineState) -> str:
    return {
        "awaiting_bronze_approval": "Bronze",
        "awaiting_silver_approval": "Silver",
        "awaiting_gold_approval": "Gold",
    }[state.status]


def _render_state(state: PipelineState) -> None:
    st.subheader("Pipeline status")
    st.write(f"**Run:** `{state.run_id}` · **Status:** `{state.status}`")
    if state.error:
        st.error(state.error)
    if state.profile_path and Path(state.profile_path).exists():
        with st.expander("Data profile"):
            st.json(json.loads(Path(state.profile_path).read_text(encoding="utf-8")))

    for title, paths in (
        ("Bronze Parquet", state.bronze_output_paths),
        ("Silver Parquet", state.silver_output_paths),
        ("Gold Parquet", state.gold_output_paths),
    ):
        if paths:
            with st.expander(title):
                for path in paths:
                    st.code(path)
                    try:
                        st.dataframe(pd.read_parquet(path).head(10), use_container_width=True)
                    except Exception as error:
                        st.warning(f"Could not preview {path}: {error}")

    if state.status in APPROVAL_STATUSES:
        label = _approval_label(state)
        st.subheader(f"Review {label} mappings")
        st.caption("Edit the generated mapping before approving it for execution.")
        sttm_path = Path(_approval_path(state))
        try:
            rules = pd.read_csv(sttm_path).fillna("")
            edited_rules = st.data_editor(
                rules,
                key=f"sttm_editor_{state.status}_{state.run_id}",
                num_rows="dynamic",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Unable to load the mapping file: {error}")
            return

        left, right = st.columns(2)
        if left.button(f"Approve {label} and continue", type="primary", key=f"approve_{state.status}"):
            edited_rules.to_csv(sttm_path, index=False)
            with st.spinner(f"Running the next pipeline phase ({label})..."):
                updated = advance_pipeline(state, approval=True)
            st.session_state["pipeline_state"] = updated
            st.rerun()
        if right.button(f"Reject {label} mapping", key=f"reject_{state.status}"):
            advance_pipeline(state, approval=False)
            st.warning("The mapping remains pending. Edit it and approve when it is ready.")
            st.rerun()

    elif state.status == "initialized":
        if st.button("Start profiling", type="primary"):
            with st.spinner("Profiling files and preparing Bronze mappings..."):
                updated = advance_pipeline(state)
            st.session_state["pipeline_state"] = updated
            st.rerun()

    elif state.status == "completed" and state.report_path:
        report_path = Path(state.report_path)
        if report_path.exists():
            st.subheader("Final report")
            report_html = report_path.read_text(encoding="utf-8")
            components.html(report_html, height=900, scrolling=True)
            st.download_button(
                "Download HTML report",
                data=report_html,
                file_name=report_path.name,
                mime="text/html",
            )


st.title("Intent-Driven Medallion Pipeline")
st.write("Upload CSV data, review each layer's mapping, and generate a report for your business question.")
if not _provider_has_key():
    st.warning(f"No API key found for the selected provider ({LLM_PROVIDER}). Add its key to .env before running agents.")

if "pipeline_state" not in st.session_state:
    st.session_state["pipeline_state"] = None

state = st.session_state["pipeline_state"]
if state is None:
    uploads = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True)
    business_intent = st.text_area("Business question", placeholder="Which product sold the most?")
    if st.button("Create pipeline run", type="primary", disabled=not uploads or not business_intent.strip()):
        run_id = str(uuid.uuid4())
        paths = _save_uploads(uploads, run_id)
        st.session_state["pipeline_state"] = create_pipeline_state(paths, business_intent.strip(), run_id)
        st.rerun()
else:
    _render_state(state)
    if st.button("Start a new run"):
        st.session_state["pipeline_state"] = None
        st.rerun()
