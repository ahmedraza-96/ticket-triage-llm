"""Response cache (ported from ahmedraza-96/ai-support-eval).

Key = sha256 over everything that can change an answer: system prompt, model,
generation config, few-shot block, ticket. Any prompt/model/config change
invalidates exactly the right entries; interrupted runs resume for free.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "cache.sqlite"


def make_key(system_prompt: str, model: str, gen_config: dict,
             few_shot_fingerprint: str, ticket: str) -> str:
    blob = json.dumps(
        {
            "prompt": system_prompt,
            "model": model,
            "gen": gen_config,
            "few_shot": few_shot_fingerprint,
            "ticket": ticket,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY,"
            " response_json TEXT NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT response_json FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO responses (key, response_json, created_at) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(response, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
