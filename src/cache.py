import sqlite3
import hashlib
import json
import os

CACHE_PATH = os.getenv("CACHE_DB_PATH", "./data/cache.db")


def _key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def _get_conn():
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def get_cached(query: str):
    try:
        conn = _get_conn()
        row = conn.execute("SELECT value FROM cache WHERE key=?", (_key(query),)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except:
        return None


def set_cache(query: str, value: dict):
    try:
        conn = _get_conn()
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                     (_key(query), json.dumps(value)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"cache write failed: {e}")