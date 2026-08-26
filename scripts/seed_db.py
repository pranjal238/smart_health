"""
FallGuard AI - Database Seeder CLI
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database.init_db import init_db

if __name__ == "__main__":
    print("Seeding FallGuard AI SQLite Database...")
    init_db()
    print("Database seeding completed.")
