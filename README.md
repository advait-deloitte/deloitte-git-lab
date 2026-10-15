# Intent-Driven Medallion Pipeline

A Streamlit application for profiling uploaded CSV files, preparing Bronze/Silver/Gold source-to-target mappings, applying human-approved transformations, and generating a DuckDB/Plotly HTML report.

## Requirements

Use Python 3.10 or newer. From PowerShell, run these commands from the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If PowerShell blocks activation, allow it only in the current session, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

Edit `.env` and set the API key for the chosen provider. Anthropic is the default; set `LLM_PROVIDER` to `anthropic`, `groq`, `gemini`, `openai`, or `github` to select another supported provider. Never commit `.env` or paste API keys into source code.

## Run the Web App

```powershell
streamlit run streamlit_app.py
```

Upload one or more CSV files and enter a business question. The pipeline pauses three times for review: Bronze mappings, Silver mappings, and Gold mappings. Review or edit each mapping table before approving it; approved rules are then applied to the data.

## Run from the CLI

```powershell
python cli.py --files "data\landing\sales.csv" --intent "Which product sold the most?"
```

The CLI prints each mapping file and waits for approval. Edit the shown CSV before approving if needed.

## Pipeline Outputs

- Profiles: `data/profiles/`
- STTM mappings: `data/sttm/`
- Bronze, Silver, and Gold Parquet data: `data/bronze_layer/`, `data/silver_layer/`, and `data/gold_layer/`
- HTML reports: `reports/`
- Audit logs: `audit_logs/`
- Agent traces: `data/traces/`
- Chroma persistent store: `.chroma/`

The current Chroma memory module stores and retrieves documents using literal text matching; it does not perform embedding-based semantic search. Groq GPT-OSS models are configured to hide reasoning output so structured tool results remain usable.

## Tests

```powershell
python -m pytest
```
