import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv(find_dotenv())

# Mapping for reasoning effort enum changes
REASONING_EFFORT_MAP = {
    "minimal": "none",
}


def init_llm(llm_args, temperature=1, max_tokens=16384):
    """Create an AzureChatOpenAI instance from a model config dict.

    The dict is copied before mutation so callers' originals are safe.
    For reasoning-style models (those with a ``reasoning_effort`` key),
    temperature is forced to 1 and max_tokens to 128 000 per Azure
    requirements.

    Args:
        llm_args: Model config dict (e.g. ``GPT54_args``).  May
            contain ``temperature``, ``max_tokens``, or
            ``max_completion_tokens`` overrides plus any
            ``AzureChatOpenAI`` kwargs.
        temperature: Default temperature if not in *llm_args*.
        max_tokens: Default max completion tokens if not in *llm_args*.

    Returns:
        An ``AzureChatOpenAI`` instance ready for use in a chain.
    """
    llm_args = llm_args.copy()
    
    # Let llm_args override defaults if present
    temperature = llm_args.pop("temperature", temperature)
    max_tokens = llm_args.pop("max_tokens", max_tokens)
    max_tokens = llm_args.pop("max_completion_tokens", max_tokens)
    
    # Force temperature = 1 for reasoning-style models
    if "reasoning_effort" in llm_args:
        original = llm_args["reasoning_effort"]
        llm_args["reasoning_effort"] = REASONING_EFFORT_MAP.get(
            original, original
        )

        temperature = 1
        max_tokens = 128000
        
    llm = AzureChatOpenAI(
        **llm_args,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
        
    return llm


GPT54m_args = dict(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    deployment_name=os.getenv("AZURE_DEPLOYMENT_GPT54M", "gpt-5.4-mini-erica"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    reasoning_effort='low'
)

GPT54_args = dict(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    deployment_name=os.getenv("AZURE_DEPLOYMENT_GPT54", "gpt-5.4-erica"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    reasoning_effort='low'
)