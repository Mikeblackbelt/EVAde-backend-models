"""Apply evade/storage/schema.sql to the D1 database named by CF_ACCOUNT_ID / CF_D1_DATABASE_ID / CF_API_TOKEN.

  python -m evade.storage.migrate
"""
from __future__ import annotations

from . import d1_from_env
from .cloudflare import SCHEMA_PATH


def main() -> int:
    n = d1_from_env().execute_script(SCHEMA_PATH.read_text(encoding="utf-8"))
    print(f"applied {n} statements from schema.sql")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
