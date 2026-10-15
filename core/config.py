import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"

for directory in (
    DATA_DIR,
    LANDING_DIR,
    PROFILES_DIR,
    STTM_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    REPORTS_DIR,
    AUDIT_DIR,
    CHROMA_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")
GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

_configured_providers = (
    ("anthropic", ANTHROPIC_API_KEY),
    ("github", GITHUB_TOKEN),
    ("openai", OPENAI_API_KEY),
    ("groq", GROQ_API_KEY),
    ("gemini", GOOGLE_API_KEY),
)
_supported_providers = {provider for provider, _ in _configured_providers}
_requested_provider = os.getenv("LLM_PROVIDER", "").strip().lower()

if _requested_provider:
    if _requested_provider not in _supported_providers:
        choices = ", ".join(sorted(_supported_providers))
        raise ValueError(f"Unsupported LLM_PROVIDER '{_requested_provider}'. Choose from: {choices}.")
    LLM_PROVIDER = _requested_provider
else:
    LLM_PROVIDER = next(
        (provider for provider, api_key in _configured_providers if api_key),
        "anthropic",
    )
