"""Manage admin-panel accounts.

Run this locally (e.g. from a terminal in the IDE) — there is intentionally no
web-facing way to create, list, or remove an account. The website can only edit a
signed-in account's own username/password (PUT /admin/profile), never provision a
new one.

Usage:
    python scripts/manage_users.py add <username> <role>
    python scripts/manage_users.py list
    python scripts/manage_users.py remove <username>
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.admin.users import ROLES, add_user, load_users, remove_user


def cmd_add(args: argparse.Namespace) -> None:
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords do not match.", file=sys.stderr)
        sys.exit(1)
    if len(password) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        sys.exit(1)

    try:
        add_user(args.username, password, args.role)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Created user '{args.username}' with role '{args.role}'.")


def cmd_list(_args: argparse.Namespace) -> None:
    users = load_users()
    if not users:
        print("No users.")
        return
    for user in users:
        print(f"{user.username}\t{user.role}")


def cmd_remove(args: argparse.Namespace) -> None:
    if remove_user(args.username):
        print(f"Removed user '{args.username}'.")
    else:
        print(f"User '{args.username}' not found.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage admin-panel accounts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Create a new account")
    add_parser.add_argument("username")
    add_parser.add_argument("role", choices=ROLES)
    add_parser.set_defaults(func=cmd_add)

    list_parser = subparsers.add_parser("list", help="List all accounts")
    list_parser.set_defaults(func=cmd_list)

    remove_parser = subparsers.add_parser("remove", help="Delete an account")
    remove_parser.add_argument("username")
    remove_parser.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
