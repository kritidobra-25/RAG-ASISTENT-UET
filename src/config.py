"""
Konfigurimi i përbashkët i projektit.

Rrugët llogariten nga vendndodhja e këtij skedari, kështu që skriptet
funksionojnë pavarësisht nga dosja nga ku ekzekutohen.
Vlerat mund të ndryshohen te skedari .env pa prekur kodin.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Dosjet
DATA_DIR = ROOT / "data"          # këtu vendosen PDF-të e UET-së
MATERIALS_DIR = DATA_DIR / "lendet"  # materialet e lëndëve (opsionale), shih data/lendet/README.md
DB_DIR = ROOT / "chroma_db"       # krijohet vetë nga build_index.py
EVAL_DIR = ROOT / "eval"          # pyetjet testuese dhe rezultatet

COLLECTION_NAME = "uet_knowledge_base"

# Modelet (OpenAI API)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Ndarja në segmente (chunking) dhe kërkimi
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))       # karaktere për segment
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # mbivendosje mes segmenteve
TOP_K = int(os.getenv("TOP_K", "5"))                    # sa segmente merren për pyetje
EMBED_BATCH_SIZE = 64                                   # segmente për një thirrje API

# Llogaritë e përdoruesve (SQLite) dhe plani mësimor i strukturuar
USERS_DB = ROOT / "users.sqlite3"                   # krijohet vetë; nuk shkon te GitHub
CURRICULUM_PATH = DATA_DIR / "curriculum.json"      # krijohet nga build_curriculum.py
