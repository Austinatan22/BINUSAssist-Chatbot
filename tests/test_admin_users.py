"""Unit tests for the pure validation helper in backend/admin/users.py.
No GPU, no Groq, no filesystem -- _validate_username is a plain regex check.
"""
import pytest

from backend.admin.users import _validate_username


class TestValidateUsername:
    @pytest.mark.parametrize("username", ["admin", "admin_2", "admin-user", "a", "A" * 64])
    def test_valid_usernames_do_not_raise(self, username):
        _validate_username(username)

    @pytest.mark.parametrize("username", ["", "admin user", "admin@user", "A" * 65, "admin!"])
    def test_invalid_usernames_raise_value_error(self, username):
        with pytest.raises(ValueError):
            _validate_username(username)
