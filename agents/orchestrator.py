import uuid
from typing import Any

from langchain_core.tools import tool

from core.agent_utils import invoke_agent
from core.audit import AuditLogger
from core.llm import get_chat_model
from core.memory import store_document
from core.state import PipelineState
from agents.bronze_agent import run_bronze_agent
from agents.gold_agent import run_gold_agent
from agents.profiler import run_profiler
from agents.reporter import run_reporter
from agents.silver_agent import run_silver_agent
from agents.sttm_generator import run_sttm_agent

APPROVAL_STATUSES = {
    "awaiting_bronze_approval",
    "awaiting_silver_approval",
    "awaiting_gold_approval",
}


def create_pipeline_state(
    uploaded_files: list[str],
    business_intent: str,
    run_id: str | None = None,
) -> PipelineState:
    identifier = run_id or str(uuid.uuid4())
    state = PipelineState(
        run_id=identifier,
        status="initialized",
        uploaded_files=uploaded_files,
        business_intent=business_intent,
        profile_path="",
        sttm_bronze_path="",
        sttm_silver_path="",
        sttm_gold_path="",
        bronze_output_paths=[],
        silver_output_paths=[],
        gold_output_paths=[],
        report_path="",
        hitl_approved=False,
    )
    AuditLogger(identifier).log(
        "orchestrator",
        "run_started",
        uploaded_files=uploaded_files,
        business_intent=business_intent,
    )
    store_document(
        f"intent-{identifier}",
        business_intent,
        {"kind": "business_intent", "run_id": identifier},
    )
    return state


def _audit(state: PipelineState, action: str, **details: Any) -> None:
    AuditLogger(state.run_id).log("orchestrator", action, status=state.status, **details)


def _require_approval(state: PipelineState, approval: bool | None) -> bool:
    if approval is None:
        raise ValueError(f"Human approval is required while status is {state.status}.")
    if not approval:
        state.hitl_approved = False
        _audit(state, "approval_rejected", sttm_path=_current_sttm_path(state))
        return False
    state.hitl_approved = True
    _audit(state, "approval_granted", sttm_path=_current_sttm_path(state))
    return True


def _current_sttm_path(state: PipelineState) -> str:
    if state.status == "awaiting_bronze_approval":
        return state.sttm_bronze_path
    if state.status == "awaiting_silver_approval":
        return state.sttm_silver_path
    if state.status == "awaiting_gold_approval":
        return state.sttm_gold_path
    return ""


def _run_supervisor_phase(
    state: PipelineState,
    phase: str,
    tools: list[Any],
    task: str,
    inputs: dict[str, Any],
    completed_steps: set[str],
    required_steps: set[str],
) -> None:
    invoke_agent(
        agent_name="supervisor",
        model=get_chat_model(),
        tools=tools,
        system_prompt=(
            "You are the pipeline Supervisor. Inspect the current phase context in the task. "
            "Call the available specialist-agent tools in the exact order requested. "
            "Wait for each result before calling the next tool, and never claim work that "
            "a tool did not complete."
        ),
        task=task,
        run_id=state.run_id,
        inputs={"phase": phase, **inputs},
    )
    missing = required_steps - completed_steps
    if missing:
        raise RuntimeError(
            f"Supervisor did not complete the required {phase} step(s): {', '.join(sorted(missing))}."
        )


def advance_pipeline(state: PipelineState, approval: bool | None = None) -> PipelineState:
    audit = AuditLogger(state.run_id)
    try:
        if state.status == "initialized":
            state.status = "profiling"
            completed_steps: set[str] = set()

            @tool
            def profiler_agent_tool() -> str:
                """Run the profiler agent and save the combined CSV profile."""
                state.profile_path = run_profiler(state.uploaded_files, state.business_intent, state.run_id)
                completed_steps.add("profile")
                return state.profile_path

            @tool
            def bronze_sttm_agent_tool() -> str:
                """Generate the Bronze source-to-target mapping after profiling."""
                if "profile" not in completed_steps:
                    raise ValueError("Run the profiler agent before generating Bronze mappings.")
                state.sttm_bronze_path = run_sttm_agent(state, "bronze")
                completed_steps.add("bronze_sttm")
                return state.sttm_bronze_path

            _run_supervisor_phase(
                state,
                "profile_and_bronze_mapping",
                [profiler_agent_tool, bronze_sttm_agent_tool],
                "Call profiler_agent_tool first, then bronze_sttm_agent_tool. "
                f"Profile these CSVs for the business intent: {state.business_intent}",
                {"uploaded_files": state.uploaded_files, "business_intent": state.business_intent},
                completed_steps,
                {"profile", "bronze_sttm"},
            )
            state.status = "awaiting_bronze_approval"
            state.hitl_approved = False
            _audit(state, "phase_complete", phase="profiling", profile_path=state.profile_path, sttm_path=state.sttm_bronze_path)
            return state

        if state.status == "awaiting_bronze_approval":
            if not _require_approval(state, approval):
                return state
            state.status = "bronze_running"
            completed_steps = set()

            @tool
            def bronze_agent_tool() -> str:
                """Apply the approved Bronze mappings and write Bronze Parquet outputs."""
                state.bronze_output_paths = run_bronze_agent(
                    state.uploaded_files,
                    state.sttm_bronze_path,
                    state.run_id,
                    "Execute the human-approved Bronze mappings for the uploaded CSV files.",
                )
                completed_steps.add("bronze")
                return str(state.bronze_output_paths)

            @tool
            def silver_sttm_agent_tool() -> str:
                """Generate Silver cleansing mappings after Bronze ingestion."""
                if "bronze" not in completed_steps:
                    raise ValueError("Run Bronze ingestion before generating Silver mappings.")
                state.sttm_silver_path = run_sttm_agent(state, "silver")
                completed_steps.add("silver_sttm")
                return state.sttm_silver_path

            _run_supervisor_phase(
                state,
                "bronze_and_silver_mapping",
                [bronze_agent_tool, silver_sttm_agent_tool],
                "Call bronze_agent_tool first to execute the approved Bronze rules, then "
                "call silver_sttm_agent_tool to draft the next mapping.",
                {"uploaded_files": state.uploaded_files, "bronze_sttm_path": state.sttm_bronze_path},
                completed_steps,
                {"bronze", "silver_sttm"},
            )
            state.status = "awaiting_silver_approval"
            state.hitl_approved = False
            _audit(state, "phase_complete", phase="bronze", output_paths=state.bronze_output_paths, sttm_path=state.sttm_silver_path)
            return state

        if state.status == "awaiting_silver_approval":
            if not _require_approval(state, approval):
                return state
            state.status = "silver_running"
            completed_steps = set()

            @tool
            def silver_agent_tool() -> str:
                """Apply approved Silver cleansing rules and write Silver Parquet outputs."""
                state.silver_output_paths = run_silver_agent(
                    state.bronze_output_paths,
                    state.sttm_silver_path,
                    state.run_id,
                    "Apply the human-approved Silver cleansing mappings to the Bronze data.",
                )
                completed_steps.add("silver")
                return str(state.silver_output_paths)

            @tool
            def gold_sttm_agent_tool() -> str:
                """Generate Gold join, grouping, and aggregation rules after Silver cleansing."""
                if "silver" not in completed_steps:
                    raise ValueError("Run Silver cleansing before generating Gold mappings.")
                state.sttm_gold_path = run_sttm_agent(state, "gold")
                completed_steps.add("gold_sttm")
                return state.sttm_gold_path

            _run_supervisor_phase(
                state,
                "silver_and_gold_mapping",
                [silver_agent_tool, gold_sttm_agent_tool],
                "Call silver_agent_tool first to apply the approved Silver rules, then "
                "call gold_sttm_agent_tool to draft Gold join and aggregation rules.",
                {"bronze_output_paths": state.bronze_output_paths, "silver_sttm_path": state.sttm_silver_path},
                completed_steps,
                {"silver", "gold_sttm"},
            )
            state.status = "awaiting_gold_approval"
            state.hitl_approved = False
            _audit(state, "phase_complete", phase="silver", output_paths=state.silver_output_paths, sttm_path=state.sttm_gold_path)
            return state

        if state.status == "awaiting_gold_approval":
            if not _require_approval(state, approval):
                return state
            state.status = "gold_running"
            completed_steps = set()

            @tool
            def gold_agent_tool() -> str:
                """Outer-join Silver tables and materialize the approved Gold tables."""
                state.gold_output_paths = run_gold_agent(
                    state.silver_output_paths,
                    state.sttm_gold_path,
                    state.business_intent,
                    state.run_id,
                    "Execute the human-approved Gold join and aggregation mappings.",
                )
                completed_steps.add("gold")
                return str(state.gold_output_paths)

            @tool
            def reporter_agent_tool() -> str:
                """Query the Gold tables and generate the final HTML report."""
                if "gold" not in completed_steps:
                    raise ValueError("Materialize Gold tables before generating the report.")
                state.report_path = run_reporter(
                    state.gold_output_paths,
                    state.business_intent,
                    state.run_id,
                    f"Analyze the Gold data and answer this business question: {state.business_intent}",
                )
                completed_steps.add("report")
                return state.report_path

            _run_supervisor_phase(
                state,
                "gold_and_reporting",
                [gold_agent_tool, reporter_agent_tool],
                "Call gold_agent_tool first to materialize approved Gold tables, then "
                "call reporter_agent_tool to answer the business question and write the report.",
                {"silver_output_paths": state.silver_output_paths, "gold_sttm_path": state.sttm_gold_path, "business_intent": state.business_intent},
                completed_steps,
                {"gold", "report"},
            )
            state.status = "completed"
            state.hitl_approved = True
            _audit(
                state,
                "run_completed",
                gold_output_paths=state.gold_output_paths,
                report_path=state.report_path,
            )
            return state

        if state.status in {"completed", "failed"}:
            return state

        raise ValueError(f"Unsupported pipeline state: {state.status}")
    except Exception as error:
        state.status = "failed"
        state.error = str(error)
        state.hitl_approved = False
        audit.log("orchestrator", "run_failed", status=state.status, error=state.error)
        return state
