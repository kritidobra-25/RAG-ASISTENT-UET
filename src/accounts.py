"""
Llogaritë dhe profilet e studentëve (Account / Login në arkitekturë).

Ruhen lokalisht te users/users.json. Fjalëkalimi nuk ruhet kurrë i hapur,
vetëm si hash scrypt me kripë të rastësishme. Skedari përjashtohet nga git.
Ky është një ruajtje e thjeshtë për demonstrim, jo një sistem autentikimi
për prodhim.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path

import config

USERS_FILE = config.ROOT / "users" / "users.json"

ROLE_ACTUAL = "aktual"
ROLE_POTENTIAL = "potencial"
ROLES = (ROLE_ACTUAL, ROLE_POTENTIAL)

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 6


class AccountError(ValueError):
    """Gabim i shfaqshëm te përdoruesi (kredenciale të pavlefshme, emër i zënë etj.)."""


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1).hex()


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(users: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def register(username: str, password: str, role: str, path: Path = USERS_FILE) -> dict:
    """Krijon një llogari të re dhe kthen të dhënat publike të saj."""
    username = username.strip()
    if not USERNAME_PATTERN.match(username):
        raise AccountError("Emri i përdoruesit duhet të ketë 3 deri 32 shkronja, shifra, _ . ose -.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(f"Fjalëkalimi duhet të ketë të paktën {MIN_PASSWORD_LENGTH} karaktere.")
    if role not in ROLES:
        raise AccountError("Zgjidh llojin e studentit.")

    users = _load(path)
    key = username.lower()
    if key in users:
        raise AccountError("Ky emër përdoruesi është i zënë.")

    salt = secrets.token_bytes(16)
    users[key] = {
        "username": username,
        "salt": salt.hex(),
        "password_hash": _hash_password(password, salt),
        "role": role,
        "profile": None,
    }
    _save(users, path)
    return public_account(users[key])


def login(username: str, password: str, path: Path = USERS_FILE) -> dict:
    """Verifikon kredencialet dhe kthen të dhënat publike të llogarisë."""
    user = _load(path).get(username.strip().lower())
    # Hash-i llogaritet edhe kur përdoruesi nuk ekziston, që koha e përgjigjes
    # të mos zbulojë cilat emra janë regjistruar.
    salt = bytes.fromhex(user["salt"]) if user else b"\0" * 16
    candidate = _hash_password(password, salt)
    if not user or not hmac.compare_digest(candidate, user["password_hash"]):
        raise AccountError("Emri i përdoruesit ose fjalëkalimi është i gabuar.")
    return public_account(user)


def save_profile(username: str, profile: dict, path: Path = USERS_FILE) -> dict:
    """Ruan profilin e studentit te llogaria dhe kthen llogarinë e përditësuar."""
    users = _load(path)
    key = username.strip().lower()
    if key not in users:
        raise AccountError("Llogaria nuk ekziston.")
    users[key]["profile"] = profile
    _save(users, path)
    return public_account(users[key])


def public_account(user: dict) -> dict:
    """Të dhënat e llogarisë pa hash-in dhe kripën."""
    return {"username": user["username"], "role": user["role"], "profile": user.get("profile")}
