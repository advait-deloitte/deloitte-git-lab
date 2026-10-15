import argparse
from pathlib import Path

from agents.orchestrator import APPROVAL_STATUSES, advance_pipeline, create_pipeline_state
from core.memory import store_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the intent-driven medallion data pipeline.")
    parser.add_argument("--files", nargs="+", required=True, help="Input CSV file paths.")
    parser.add_argument("--intent", required=True, help="Business question in plain English.")
    args = parser.parse_args()

    missing = [path for path in args.files if not Path(path).is_file()]
    if missing:
        parser.error(f"Input file(s) do not exist: {', '.join(missing)}")

    state = create_pipeline_state(args.files, args.intent)
    store_document(
        f"intent-cli-{state.run_id}",
        args.intent,
        {"kind": "business_intent", "run_id": state.run_id, "source": "cli"},
    )

    while state.status not in {"completed", "failed"}:
        if state.status not in APPROVAL_STATUSES:
            state = advance_pipeline(state)
        else:
            sttm_path = {
                "awaiting_bronze_approval": state.sttm_bronze_path,
                "awaiting_silver_approval": state.sttm_silver_path,
                "awaiting_gold_approval": state.sttm_gold_path,
            }[state.status]
            print(f"\nReview the generated mapping: {sttm_path}")
            print(Path(sttm_path).read_text(encoding="utf-8"))
            response = input("Approve this mapping? Edit the CSV first if needed. [y/N]: ").strip().casefold()
            state = advance_pipeline(state, approval=response in {"y", "yes"})

        print(f"Run {state.run_id} status: {state.status}")
        if state.error:
            print(f"Error: {state.error}")
            break

    if state.status == "completed":
        print(f"Pipeline completed. HTML report: {state.report_path}")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
