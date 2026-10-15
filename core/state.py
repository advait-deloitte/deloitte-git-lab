from typing import Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    run_id: str = Field(default="", description="Unique identifier for this pipeline run.")
    status: str = Field(default="initialized", description="Current status of the pipeline run.")

    uploaded_files: list[str] = Field(
        default_factory=list,
        description="Paths to uploaded CSV files.",
    )
    business_intent: str = Field(
        default="",
        description="User's question or request in plain English.",
    )

    profile_path: str = Field(
        default="",
        description="Path to the generated data profile JSON file.",
    )

    sttm_bronze_path: str = Field(
        ...,
        description="Path to the source-to-target mapping rules for the Bronze layer.",
    )
    sttm_silver_path: str = Field(
        ...,
        description="Path to the source-to-target mapping rules for the Silver layer.",
    )
    sttm_gold_path: str = Field(
        ...,
        description="Path to the source-to-target mapping rules for the Gold layer.",
    )
    hitl_approved: bool = Field(
        default=False,
        description="Whether a human has approved the mapping rules.",
    )

    bronze_output_paths: list[str] = Field(
        ...,
        description="Paths to the generated Bronze Parquet files.",
    )
    silver_output_paths: list[str] = Field(
        ...,
        description="Paths to the generated Silver Parquet files.",
    )
    gold_output_paths: list[str] = Field(
        ...,
        description="Paths to the generated Gold Parquet files.",
    )

    report_path: str = Field(
        default="",
        description="Path to the final HTML report.",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if the pipeline run failed.",
    )
