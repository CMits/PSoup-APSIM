# lib/engage_db.py
from __future__ import annotations
import sqlite3, os, time
from contextlib import contextmanager
from typing import Optional, List, Tuple, Dict, Any

DB_PATH = os.environ.get("ENGAGE_DB_PATH", "engage.db")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS ideas (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT UNIQUE NOT NULL,            -- stable string like 'psoup-apsim-connector'
  name TEXT NOT NULL,
  description TEXT,
  created_by TEXT,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS votes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_id INTEGER NOT NULL,
  voter TEXT NOT NULL,                 -- free text (name/email) or "anonymous"
  vote INTEGER NOT NULL,               -- 1=yes, 0=unsure, -1=no
  confidence INTEGER DEFAULT 50,       -- 0-100
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(idea_id, voter),
  FOREIGN KEY (idea_id) REFERENCES ideas(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_id INTEGER NOT NULL,
  author TEXT NOT NULL,
  text TEXT NOT NULL,
  tag TEXT,                            -- 'scientific', 'practical', 'presentation', 'implementation', etc.
  anonymous INTEGER DEFAULT 0,         -- 0/1
  parent_id INTEGER,                   -- for threads
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (idea_id) REFERENCES ideas(id) ON DELETE CASCADE,
  FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS comment_votes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  comment_id INTEGER NOT NULL,
  voter TEXT NOT NULL,
  direction INTEGER NOT NULL,          -- 1 = upvote, -1 = downvote (you’ll likely only use +1)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(comment_id, voter),
  FOREIGN KEY (comment_id) REFERENCES comments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_id INTEGER NOT NULL,
  title TEXT NOT NULL,
  owner TEXT,                          -- optional assignee
  due_date TEXT,                       -- ISO date string
  done INTEGER DEFAULT 0,              -- 0/1
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  closed_at TIMESTAMP,
  FOREIGN KEY (idea_id) REFERENCES ideas(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_id INTEGER NOT NULL,
  version_label TEXT NOT NULL,         -- e.g. 'v0.3'
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (idea_id) REFERENCES ideas(id) ON DELETE CASCADE
);
"""

@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

# --- init ---
def init_db():
    with connect() as con:
        con.executescript(SCHEMA)

# --- ideas ---
def upsert_idea(key: str, name: str, description: str = "", created_by: str = "") -> int:
    with connect() as con:
        cur = con.cursor()
        cur.execute("SELECT id FROM ideas WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE ideas SET name=?, description=? WHERE id=?", (name, description, row["id"]))
            return row["id"]
        cur.execute("INSERT INTO ideas(key, name, description, created_by) VALUES(?,?,?,?)",
                    (key, name, description, created_by))
        return cur.lastrowid

def get_idea_by_key(key: str) -> Optional[sqlite3.Row]:
    with connect() as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM ideas WHERE key=?", (key,))
        return cur.fetchone()

def list_ideas() -> List[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM ideas ORDER BY created_at DESC").fetchall()

# --- votes ---
def cast_vote(idea_id: int, voter: str, vote: int, confidence: int):
    with connect() as con:
        con.execute("""
            INSERT INTO votes(idea_id, voter, vote, confidence)
            VALUES(?,?,?,?)
            ON CONFLICT(idea_id, voter) DO UPDATE SET vote=excluded.vote, confidence=excluded.confidence, created_at=CURRENT_TIMESTAMP
        """, (idea_id, voter, vote, confidence))

def fetch_votes(idea_id: int) -> List[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM votes WHERE idea_id=? ORDER BY created_at DESC", (idea_id,)).fetchall()

# --- comments ---
def add_comment(idea_id: int, author: str, text: str, tag: str, anonymous: bool, parent_id: Optional[int]):
    with connect() as con:
        con.execute("""
            INSERT INTO comments(idea_id, author, text, tag, anonymous, parent_id)
            VALUES(?,?,?,?,?,?)
        """, (idea_id, author, text, tag, int(anonymous), parent_id))

def fetch_comments(idea_id: int) -> List[sqlite3.Row]:
    with connect() as con:
        return con.execute("""
            SELECT c.*, 
                   COALESCE(SUM(cv.direction), 0) AS score
            FROM comments c
            LEFT JOIN comment_votes cv ON cv.comment_id = c.id
            WHERE c.idea_id=?
            GROUP BY c.id
            ORDER BY (CASE WHEN c.parent_id IS NULL THEN 0 ELSE 1 END), score DESC, c.created_at DESC
        """, (idea_id,)).fetchall()

def vote_comment(comment_id: int, voter: str, direction: int = 1):
    with connect() as con:
        con.execute("""
            INSERT INTO comment_votes(comment_id, voter, direction)
            VALUES(?,?,?)
            ON CONFLICT(comment_id, voter) DO UPDATE SET direction=excluded.direction, created_at=CURRENT_TIMESTAMP
        """, (comment_id, voter, direction))

# --- actions ---
def add_action(idea_id: int, title: str, owner: str = "", due_date: Optional[str] = None):
    with connect() as con:
        con.execute("INSERT INTO actions(idea_id, title, owner, due_date) VALUES(?,?,?,?)",
                    (idea_id, title, owner, due_date))

def set_action_done(action_id: int, done: bool):
    with connect() as con:
        if done:
            con.execute("UPDATE actions SET done=1, closed_at=CURRENT_TIMESTAMP WHERE id=?", (action_id,))
        else:
            con.execute("UPDATE actions SET done=0, closed_at=NULL WHERE id=?", (action_id,))

def fetch_actions(idea_id: int) -> List[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM actions WHERE idea_id=? ORDER BY done, created_at DESC", (idea_id,)).fetchall()

# --- versions ---
def add_version(idea_id: int, label: str, notes: str = ""):
    with connect() as con:
        con.execute("INSERT INTO versions(idea_id, version_label, notes) VALUES(?,?,?)", (idea_id, label, notes))

def fetch_versions(idea_id: int) -> List[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM versions WHERE idea_id=? ORDER BY created_at DESC", (idea_id,)).fetchall()
