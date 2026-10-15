from typing import Any

from core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def get_chat_model(temperature: float = 0) -> Any:
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("Set ANTHROPIC_API_KEY in .env to use the Anthropic provider.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("Set GROQ_API_KEY in .env to use the Groq provider.")
        from langchain_groq import ChatGroq

        options: dict[str, Any] = {
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "temperature": temperature,
        }
        if GROQ_MODEL.startswith("openai/gpt-oss-"):
            options["reasoning_format"] = "hidden"
        return ChatGroq(**options)

    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("Set GOOGLE_API_KEY in .env to use the Gemini provider.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY in .env to use the OpenAI provider.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=temperature)

    if LLM_PROVIDER == "github":
        if not GITHUB_TOKEN:
            raise ValueError("Set GITHUB_TOKEN in .env to use GitHub Models.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=GITHUB_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_BASE_URL,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")