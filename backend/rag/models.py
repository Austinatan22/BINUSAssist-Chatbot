import logging

import torch
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from backend.config import settings as app_settings
from backend.rag.token_budget import get_token_counter
from backend.state import app_state

logger = logging.getLogger(__name__)

_initialized = False


def _build_llm():
    """Construct the generation LLM for the configured provider (config.llm_provider).
    Downstream everything uses LlamaIndex's global Settings.llm identically -- only the
    vendor/model differs; retrieval and reranking are unaffected either way."""
    provider = app_settings.llm_provider.lower()
    if provider == "gemini":
        # Imported lazily so a Groq-only deployment doesn't require the google-genai stack.
        from google.genai import types
        from llama_index.llms.google_genai import GoogleGenAI

        logger.info("Configuring Gemini LLM %s", app_settings.gemini_model)
        # A fully-formed GenerateContentConfig REPLACES the temperature/max_tokens the
        # integration would otherwise build (see its base.py), so every generation param
        # must live inside it. thinking_config is sent ONLY when gemini_thinking_budget is
        # set -- flash-lite (the default) has thinking off already and 400s if the field is
        # present; a full-flash tier needs budget=0 to disable its reasoning pass (thinking
        # adds seconds + eats the token cap for no benefit on grounded RAG). No
        # frequency_penalty -- gemini rejects it ("Penalty is not enabled for this model"),
        # and it was only a mild repetition guard for the weaker 8B model anyway.
        gen_kwargs = dict(
            temperature=app_settings.llm_temperature,
            max_output_tokens=app_settings.llm_max_tokens,
        )
        if app_settings.gemini_thinking_budget is not None:
            gen_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=app_settings.gemini_thinking_budget
            )
        generation_config = types.GenerateContentConfig(**gen_kwargs)
        return GoogleGenAI(
            model=app_settings.gemini_model,
            api_key=app_settings.gemini_api_key,
            # metadata/token-budget bookkeeping only; generation_config drives the request.
            max_tokens=app_settings.llm_max_tokens,
            generation_config=generation_config,
        )

    if provider == "openai":
        # OpenAI's own models via the same OpenAI-compatible client Groq subclasses -- so
        # the config mirrors the Groq branch exactly (frequency_penalty is a native OpenAI
        # request field, passed through additional_kwargs). reasoning_effort is forwarded
        # only when set (meaningful for o-series / gpt-5 reasoning models, ignored by 4o-mini).
        from llama_index.llms.openai import OpenAI

        logger.info("Configuring OpenAI LLM %s", app_settings.openai_model)
        additional_kwargs = {"frequency_penalty": app_settings.llm_frequency_penalty}
        if app_settings.llm_reasoning_effort is not None:
            additional_kwargs["reasoning_effort"] = app_settings.llm_reasoning_effort
        return OpenAI(
            model=app_settings.openai_model,
            api_key=app_settings.openai_api_key,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            additional_kwargs=additional_kwargs,
        )

    logger.info("Configuring Groq LLM %s", app_settings.llm_model)
    # frequency_penalty isn't a first-class llama-index constructor param (unlike
    # max_tokens/temperature) -- additional_kwargs is the path that reaches the request
    # body regardless of model name, same reason reasoning_effort below uses it.
    additional_kwargs = {"frequency_penalty": app_settings.llm_frequency_penalty}
    if app_settings.llm_reasoning_effort is not None:
        additional_kwargs["reasoning_effort"] = app_settings.llm_reasoning_effort
    return Groq(
        model=app_settings.llm_model,
        api_key=app_settings.groq_api_key,
        temperature=app_settings.llm_temperature,
        max_tokens=app_settings.llm_max_tokens,
        additional_kwargs=additional_kwargs,
    )


def init_models() -> None:
    """Load the embedding model and configure the LLM once, into LlamaIndex global Settings."""
    global _initialized
    if _initialized:
        return

    device = app_settings.embedding_device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU for embeddings")
        device = "cpu"
    # Actual (post-fallback) device, not just the configured one -- /health reports this
    # so a silent CPU fallback (10-15x slower embedding) is visible without reading logs.
    app_state["embedding_device"] = device

    logger.info("Loading embedding model %s on %s", app_settings.embedding_model_name, device)
    # fp16 weights on CUDA: bge-m3 is a ~568M-param XLM-RoBERTa-large encoder; loading it
    # in half precision cut its resident VRAM from 2166MiB to 1083MiB (measured directly on
    # this project's RTX 3080), an exact halving and ~1.06GB reclaimed, with NO meaningful
    # retrieval-quality cost -- fp32-vs-fp16 embeddings of the same text are 0.999999 cosine
    # identical and their pairwise-similarity matrix differs by at most 1.6e-4, so ranking
    # is preserved. Same tradeoff already validated for the cross-encoder reranker (see
    # build_reranker's .half()). Loaded directly in fp16 via `dtype` (threaded through
    # HuggingFaceEmbedding -> SentenceTransformer(model_kwargs=...) -> AutoModel.from_pretrained)
    # rather than an in-place .half() after load, so peak load-time memory is fp16 too, not
    # a momentary fp32 spike. CUDA-only: fp16 matmul on CPU is slow/poorly-supported, and
    # the embedding_device already falls back to CPU above when CUDA is unavailable.
    embed_kwargs = {}
    if device == "cuda":
        embed_kwargs["model_kwargs"] = {"dtype": torch.float16}
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=app_settings.embedding_model_name,
        device=device,
        embed_batch_size=app_settings.embedding_batch_size,
        **embed_kwargs,
    )

    Settings.llm = _build_llm()

    # Daily token budget (IMPROVEMENTS.md #3.2): observes every real Groq call through
    # the shared Settings.llm instance -- see backend/rag/token_budget.py for why.
    Settings.callback_manager = CallbackManager([get_token_counter()])

    _initialized = True
