import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent / "resume_analyzer.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_coach_tables():
    """Anonymous feedback tables for the Career & Interview Coach (no CV text is stored)."""
    conn = get_db_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS coach_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                reason TEXT,
                focus TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS coach_challenges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT NOT NULL,
                category TEXT NOT NULL,
                reason TEXT NOT NULL,
                focus TEXT,
                status TEXT DEFAULT 'open',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    finally:
        conn.close()
def add_missing_job_columns():
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("PRAGMA table_info(jobs)")
        existing_columns = [
            row["name"] for row in cursor.fetchall()
        ]

        new_columns = {
            "location": "TEXT",
            "salary": "TEXT",
            "job_type": "TEXT"
        }

        for column, column_type in new_columns.items():
            if column not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column} {column_type}"
                )

        connection.commit()

    finally:
        connection.close()