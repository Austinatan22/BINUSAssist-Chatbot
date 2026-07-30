"""Soft daily token budget (IMPROVEMENTS.md #3.2).

A provider's daily token limit is easy to exhaust -- Groq's TPD went from eval traffic
alone during development (see PROJECT_LOG.md) -- and nothing in production stops real
traffic, or abuse, from doing the same and taking the bot down for the rest of the day
with a hard provider error. Provider-agnostic: it caps whatever LLM is configured.

Wraps llama-index's built-in TokenCountingHandler, attached to Settings.callback_manager
in backend/rag/models.py so it observes every real Settings.llm call across the process
(condense_question, the named-program classifier, paraphrase rewriting, and the main
generation call all share the one Settings.llm instance). It reads actual usage off each
LLM response when present, falling back to a tokenizer estimate only if a response
lacks a usage field -- not a guess-based heuristic.

Deliberately NOT using the handler's own built-in `token_budget` param: that raises a
ValueError from on_event_start on EVERY event (including embedding lookups, which must
keep working for the semantic cache even once the LLM budget is spent) with no
day-of-week reset. Instead this module only reads total_llm_token_count and gates one
explicit call site (backend/main.py, right before the generation call and the
paraphrase-retry call) -- both LLM-token-spending, neither embeddings/retrieval.

A cache hit (backend/rag/cache.py) never reaches Settings.llm at all, so it keeps
working even at zero remaining budget -- the "serve cached content" half of this item's
graceful-degradation ask. The other half (declining fresh generation instead of a hard
provider failure) is is_budget_exceeded() below.
"""
from datetime import datetime, timezone

from llama_index.core.callbacks import TokenCountingHandler

from backend.config import settings

_token_counter = TokenCountingHandler()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# Seeded to the day the process started, not None -- otherwise the first-ever call to
# is_budget_exceeded() would see _current_day != today and wipe out any usage already
# recorded before that first check, mistaking process startup for a day rollover.
_current_day: str = _today()


def get_token_counter() -> TokenCountingHandler:
    """The single shared handler instance -- attach this to Settings.callback_manager
    once, in init_models(), rather than constructing a second one here."""
    return _token_counter


def _roll_over_if_new_day() -> None:
    global _current_day
    today = _today()
    if _current_day != today:
        _current_day = today
        _token_counter.reset_counts()


def is_budget_exceeded() -> bool:
    """True once today's cumulative LLM token usage (prompt + completion, across
    every request this process has served today) has crossed settings.daily_token_budget.
    A budget of 0 disables the check entirely (always False)."""
    if not settings.daily_token_budget:
        return False
    _roll_over_if_new_day()
    return _token_counter.total_llm_token_count >= settings.daily_token_budget
