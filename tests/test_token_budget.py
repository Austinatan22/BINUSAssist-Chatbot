"""Unit tests for backend/rag/token_budget.py (IMPROVEMENTS.md #3.2). No Groq, no GPU --
TokenCountingHandler is a plain in-memory counter; these tests feed it synthetic
TokenCountingEvent records directly instead of making a real LLM call.
"""
import pytest
from llama_index.core.callbacks.token_counting import TokenCountingEvent

from backend.config import settings
from backend.rag import token_budget


@pytest.fixture(autouse=True)
def _reset_state():
    token_budget.get_token_counter().reset_counts()
    token_budget._current_day = token_budget._today()
    yield
    token_budget.get_token_counter().reset_counts()
    token_budget._current_day = token_budget._today()


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    token_budget.get_token_counter().llm_token_counts.append(
        TokenCountingEvent(
            prompt="p", completion="c",
            prompt_token_count=prompt_tokens, completion_token_count=completion_tokens,
        )
    )


class TestIsBudgetExceeded:
    def test_budget_of_zero_disables_the_check(self, monkeypatch):
        monkeypatch.setattr(settings, "daily_token_budget", 0)
        _record_usage(10**9, 10**9)
        assert token_budget.is_budget_exceeded() is False

    def test_false_when_under_budget(self, monkeypatch):
        monkeypatch.setattr(settings, "daily_token_budget", 1000)
        _record_usage(100, 100)
        assert token_budget.is_budget_exceeded() is False

    def test_true_once_at_or_over_budget(self, monkeypatch):
        monkeypatch.setattr(settings, "daily_token_budget", 1000)
        _record_usage(500, 600)
        assert token_budget.is_budget_exceeded() is True

    def test_usage_accumulates_across_multiple_calls(self, monkeypatch):
        monkeypatch.setattr(settings, "daily_token_budget", 1000)
        _record_usage(400, 400)
        assert token_budget.is_budget_exceeded() is False
        _record_usage(100, 200)
        assert token_budget.is_budget_exceeded() is True

    def test_new_calendar_day_resets_accumulated_usage(self, monkeypatch):
        monkeypatch.setattr(settings, "daily_token_budget", 1000)
        _record_usage(500, 600)
        assert token_budget.is_budget_exceeded() is True

        monkeypatch.setattr(token_budget, "_today", lambda: "2099-01-01")
        assert token_budget.is_budget_exceeded() is False
