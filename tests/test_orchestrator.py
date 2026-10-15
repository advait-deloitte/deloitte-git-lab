from agents import orchestrator
from core import audit


def test_supervisor_runs_phases_in_order_with_approval_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(orchestrator, "store_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "get_chat_model", lambda: object())
    monkeypatch.setattr(
        orchestrator,
        "invoke_agent",
        lambda **kwargs: [tool.invoke({}) for tool in kwargs["tools"]],
    )
    monkeypatch.setattr(orchestrator, "run_profiler", lambda *args: "profile.json")
    monkeypatch.setattr(orchestrator, "run_sttm_agent", lambda state, layer: f"{layer}.csv")
    monkeypatch.setattr(orchestrator, "run_bronze_agent", lambda *args: ["bronze.parquet"])
    monkeypatch.setattr(orchestrator, "run_silver_agent", lambda *args: ["silver.parquet"])
    monkeypatch.setattr(orchestrator, "run_gold_agent", lambda *args: ["gold.parquet"])
    monkeypatch.setattr(orchestrator, "run_reporter", lambda *args: "report.html")

    state = orchestrator.create_pipeline_state(["sales.csv"], "Which product sold most?", "run-12345678")
    state = orchestrator.advance_pipeline(state)
    assert state.status == "awaiting_bronze_approval"
    assert state.profile_path == "profile.json"

    rejected = orchestrator.advance_pipeline(state, approval=False)
    assert rejected.status == "awaiting_bronze_approval"
    assert not rejected.hitl_approved

    state = orchestrator.advance_pipeline(state, approval=True)
    assert state.status == "awaiting_silver_approval"
    assert state.bronze_output_paths == ["bronze.parquet"]

    state = orchestrator.advance_pipeline(state, approval=True)
    assert state.status == "awaiting_gold_approval"
    assert state.silver_output_paths == ["silver.parquet"]

    state = orchestrator.advance_pipeline(state, approval=True)
    assert state.status == "completed"
    assert state.gold_output_paths == ["gold.parquet"]
    assert state.report_path == "report.html"
