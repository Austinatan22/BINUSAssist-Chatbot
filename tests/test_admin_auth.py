"""Unit tests for the brute-force lockout helpers in backend/admin/auth.py
(IMPROVEMENTS.md #8.1). Pure logic against an injected `now` -- no real clock, no
FastAPI request/response plumbing needed to exercise the lockout state machine itself.
"""
import pytest

from backend.admin import auth


@pytest.fixture(autouse=True)
def _reset_state():
    auth._failed_attempts.clear()
    auth._locked_until.clear()
    yield
    auth._failed_attempts.clear()
    auth._locked_until.clear()


class TestLockout:
    def test_not_locked_out_with_no_history(self):
        assert auth._seconds_locked_out("1.2.3.4", now=1000.0) == 0.0

    def test_stays_unlocked_below_the_failure_threshold(self):
        for i in range(auth._MAX_FAILED_ATTEMPTS - 1):
            auth._record_failed_attempt("1.2.3.4", now=1000.0 + i)
        assert auth._seconds_locked_out("1.2.3.4", now=1000.0) == 0.0

    def test_locks_out_once_the_threshold_is_reached(self):
        for i in range(auth._MAX_FAILED_ATTEMPTS):
            auth._record_failed_attempt("1.2.3.4", now=1000.0 + i)
        remaining = auth._seconds_locked_out("1.2.3.4", now=1000.0 + auth._MAX_FAILED_ATTEMPTS)
        assert remaining > 0

    def test_lockout_expires_after_the_lockout_window(self):
        for i in range(auth._MAX_FAILED_ATTEMPTS):
            auth._record_failed_attempt("1.2.3.4", now=1000.0 + i)
        far_future = 1000.0 + auth._LOCKOUT_SECONDS + auth._MAX_FAILED_ATTEMPTS + 1
        assert auth._seconds_locked_out("1.2.3.4", now=far_future) == 0.0

    def test_old_attempts_outside_the_window_do_not_count(self):
        auth._record_failed_attempt("1.2.3.4", now=1000.0)
        # A gap longer than the attempt window before the rest of the attempts land --
        # the first one should have aged out and no longer count toward the threshold.
        later = 1000.0 + auth._ATTEMPT_WINDOW_SECONDS + 1
        for i in range(auth._MAX_FAILED_ATTEMPTS - 1):
            auth._record_failed_attempt("1.2.3.4", now=later + i)
        assert auth._seconds_locked_out("1.2.3.4", now=later) == 0.0

    def test_clear_failed_attempts_removes_the_lockout(self):
        for i in range(auth._MAX_FAILED_ATTEMPTS):
            auth._record_failed_attempt("1.2.3.4", now=1000.0 + i)
        auth._clear_failed_attempts("1.2.3.4")
        assert auth._seconds_locked_out("1.2.3.4", now=1000.0 + auth._MAX_FAILED_ATTEMPTS) == 0.0

    def test_lockout_is_scoped_per_key(self):
        for i in range(auth._MAX_FAILED_ATTEMPTS):
            auth._record_failed_attempt("1.2.3.4", now=1000.0 + i)
        assert auth._seconds_locked_out("5.6.7.8", now=1000.0 + auth._MAX_FAILED_ATTEMPTS) == 0.0
