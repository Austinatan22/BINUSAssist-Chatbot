import base64
import binascii
import math
import time

import bcrypt
from fastapi import Depends, Header, HTTPException, Request, status
from slowapi.util import get_remote_address

from backend.admin.users import User, find_user, verify_password

# Brute-force lockout on admin auth (IMPROVEMENTS.md #8.1) -- HTTP Basic + sessionStorage
# has no lockout or rate limit on failed auth today, and the /chat limiter
# (backend/main.py) doesn't cover this router. In-memory, per-process, keyed by client
# IP -- same "fine for a single-process prototype" tradeoff as that limiter (resets on
# restart, doesn't share state across multiple workers/instances). Keyed by IP rather
# than attempted username so a locked-out attacker can't use the lockout itself to probe
# which usernames exist.
_MAX_FAILED_ATTEMPTS = 5
_ATTEMPT_WINDOW_SECONDS = 15 * 60
_LOCKOUT_SECONDS = 15 * 60

_failed_attempts: dict[str, list[float]] = {}
_locked_until: dict[str, float] = {}

# A stable dummy hash to check the submitted password against when the username doesn't
# exist, so a bad password against an unknown username takes roughly the same time as a
# bad password against a real one -- otherwise the bcrypt-comparison time difference
# would let an attacker enumerate valid usernames.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())


def _seconds_locked_out(key: str, now: float) -> float:
    """Seconds remaining until key's lockout expires, or 0 if not currently locked out."""
    locked_until = _locked_until.get(key)
    if locked_until is None or now >= locked_until:
        return 0.0
    return locked_until - now


def _record_failed_attempt(key: str, now: float) -> None:
    attempts = [t for t in _failed_attempts.get(key, []) if now - t < _ATTEMPT_WINDOW_SECONDS]
    attempts.append(now)
    _failed_attempts[key] = attempts
    if len(attempts) >= _MAX_FAILED_ATTEMPTS:
        _locked_until[key] = now + _LOCKOUT_SECONDS


def _clear_failed_attempts(key: str) -> None:
    _failed_attempts.pop(key, None)
    _locked_until.pop(key, None)


def get_current_user(request: Request, authorization: str = Header(default="")) -> User:
    """Parses HTTP Basic auth (Authorization: Basic base64(username:password)) and looks
    the account up in the file-backed user store."""
    key = get_remote_address(request)
    now = time.monotonic()

    remaining = _seconds_locked_out(key, now)
    if remaining > 0:
        retry_after = math.ceil(remaining)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    scheme, _, encoded = authorization.partition(" ")
    user = None
    if scheme == "Basic" and encoded:
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            decoded = ""
        username, _, password = decoded.partition(":")
        candidate = find_user(username)
        if candidate is not None:
            if verify_password(candidate, password):
                user = candidate
        else:
            bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)

    if user is None:
        _record_failed_attempt(key, now)
        # No WWW-Authenticate header: the frontend has its own login form and never
        # wants the browser's native Basic-auth credential popup triggered, which
        # happens automatically (independent of fetch/XHR) any time a response carries
        # this header.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    _clear_failed_attempts(key)
    return user


def require_role(*roles: str):
    """Dependency factory: only the only role today is "admin", but routes can require
    any subset of ROLES once more roles exist, without changing how auth itself works."""

    def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return _check


require_admin = require_role("admin")
