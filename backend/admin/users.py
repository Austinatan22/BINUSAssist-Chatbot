"""File-backed account store for the admin panel.

Accounts can only be created, listed, or removed via scripts/manage_users.py, run
locally (e.g. from a terminal in the IDE) — there is intentionally no API endpoint
that creates an account, so a new admin can never be provisioned from the website.
"""

import json
import re

import bcrypt
from pydantic import BaseModel

from backend.config import settings

ROLES = ("admin",)

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class User(BaseModel):
    username: str
    password_hash: str
    role: str


def _validate_username(username: str) -> None:
    if not _USERNAME_RE.match(username):
        raise ValueError(
            "Username must be 1-64 characters using only letters, numbers, '_', or '-'"
        )


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def load_users() -> list[User]:
    if not settings.users_path.exists():
        return []
    raw = json.loads(settings.users_path.read_text(encoding="utf-8"))
    return [User(**entry) for entry in raw]


def save_users(users: list[User]) -> None:
    settings.users_path.write_text(
        json.dumps([u.model_dump() for u in users], indent=2), encoding="utf-8"
    )


def find_user(username: str) -> User | None:
    for user in load_users():
        if user.username == username:
            return user
    return None


def verify_password(user: User, password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8"))


def add_user(username: str, password: str, role: str) -> User:
    _validate_username(username)
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}'. Valid roles: {', '.join(ROLES)}")
    if find_user(username) is not None:
        raise ValueError(f"User '{username}' already exists")

    user = User(username=username, password_hash=_hash_password(password), role=role)
    users = load_users()
    users.append(user)
    save_users(users)
    return user


def remove_user(username: str) -> bool:
    users = load_users()
    remaining = [u for u in users if u.username != username]
    if len(remaining) == len(users):
        return False
    save_users(remaining)
    return True


def update_user(
    username: str, *, new_username: str | None = None, new_password: str | None = None
) -> User:
    """Updates a user's own username/password — used by the profile editor, not account creation."""
    if new_username is not None:
        _validate_username(new_username)

    users = load_users()
    for i, user in enumerate(users):
        if user.username != username:
            continue
        if new_username and new_username != username and find_user(new_username) is not None:
            raise ValueError(f"User '{new_username}' already exists")

        updated = user.model_copy(
            update={
                "username": new_username or user.username,
                "password_hash": _hash_password(new_password) if new_password else user.password_hash,
            }
        )
        users[i] = updated
        save_users(users)
        return updated

    raise ValueError(f"User '{username}' not found")
