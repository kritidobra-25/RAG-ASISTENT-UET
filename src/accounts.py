"""
Llogaritë e përdoruesve (SQLite): regjistrimi, hyrja dhe ruajtja e profilit.

Fjalëkalimet ruhen vetëm si hash PBKDF2-SHA256 me kripë të rastësishme. Përdoret
emër përdoruesi (pseudonim), jo emri real ose email. Biseda NUK ruhet: vetëm
roli, profili i studentit dhe të dhënat akademike.
"""

import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone

import config
from student_profile import new_academic, new_profile

ROLES = {"current": "Student aktual", "prospective": "Student potencial"}
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
_ITERATIONS = 200_000


class AccountError(ValueError):
    """Gabim i shfaqshëm te përdoruesi (në shqip)."""


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(config.USERS_DB)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with closing(_connect()) as db, db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                profile TEXT NOT NULL,
                academic TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS).hex()


def _to_user(row: sqlite3.Row) -> dict:
    profile = {**new_profile(), **json.loads(row["profile"])}
    academic = {**new_academic(), **json.loads(row["academic"])}
    return {"id": row["id"], "username": row["username"], "role": row["role"], "profile": profile, "academic": academic}


def create_user(username: str, password: str, role: str) -> dict:
    username = username.strip()
    if not _USERNAME.match(username):
        raise AccountError("Emri i përdoruesit duhet të ketë 3 deri 30 shkronja, numra ose . _ -")
    if len(password) < 8:
        raise AccountError("Fjalëkalimi duhet të ketë të paktën 8 karaktere.")
    if role not in ROLES:
        raise AccountError("Zgjidh llojin e studentit.")
    salt = os.urandom(16)
    try:
        with closing(_connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO users (username, password_hash, salt, role, profile, academic, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    username,
                    _hash(password, salt),
                    salt.hex(),
                    role,
                    json.dumps(new_profile()),
                    json.dumps(new_academic()),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.IntegrityError as error:
        raise AccountError("Ky emër përdoruesi është i zënë.") from error
    return load_user(cursor.lastrowid)


def authenticate(username: str, password: str) -> dict | None:
    """Kthen përdoruesin nëse të dhënat janë të sakta, përndryshe None."""
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    # Hash-i llogaritet edhe kur përdoruesi nuk ekziston, që koha e përgjigjes të mos e zbulojë.
    salt = bytes.fromhex(row["salt"]) if row else b"\x00" * 16
    candidate = _hash(password, salt)
    if row and hmac.compare_digest(candidate, row["password_hash"]):
        return _to_user(row)
    time.sleep(0.5)  # ngadalëson provat e njëpasnjëshme
    return None


def load_user(user_id: int) -> dict:
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _to_user(row)


def save_user_data(user_id: int, profile: dict, academic: dict, role: str) -> None:
    with closing(_connect()) as db, db:
        db.execute(
            "UPDATE users SET profile = ?, academic = ?, role = ? WHERE id = ?",
            (json.dumps(profile, ensure_ascii=False), json.dumps(academic, ensure_ascii=False), role, user_id),
        )
