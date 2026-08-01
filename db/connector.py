"""MySQL connection helpers for RetORM."""

import sys
from typing import Any, Dict, List, Optional

import pymysql
import pymysql.cursors
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, __file__.rsplit("/db", 1)[0])
import config

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_connection() -> pymysql.connections.Connection:
    cfg = config.DB_CONFIG
    return pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg["charset"],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def execute_sql(sql: str, params=None) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            return list(cursor.fetchall())
    finally:
        conn.close()


def execute_many(sql: str, data: List[tuple]) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(sql, data)
    finally:
        conn.close()


def init_database() -> None:
    cfg = config.DB_CONFIG
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        charset=cfg["charset"],
        autocommit=True,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        print(f"[connector] database `{cfg['database']}` ready.")
    finally:
        conn.close()


def create_tables(ddl_statements: List[str]) -> None:
    for ddl in ddl_statements:
        execute_sql(ddl)
        print(f"[connector] table `{_extract_table_name(ddl)}` created.")


def drop_tables(table_names: List[str]) -> None:
    for name in table_names:
        execute_sql(f"DROP TABLE IF EXISTS `{name}`;")
        print(f"[connector] table `{name}` dropped.")


def truncate_tables(table_names: List[str]) -> None:
    for name in table_names:
        execute_sql(f"TRUNCATE TABLE `{name}`;")


def insert_rows(table: str, columns: List[str], rows: List[tuple]) -> None:
    if not rows:
        return
    col_str = ", ".join(f"`{c}`" for c in columns)
    placeholder_str = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO `{table}` ({col_str}) VALUES ({placeholder_str});"
    execute_many(sql, rows)
    print(f"[connector] inserted {len(rows)} rows into `{table}`.")


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        cfg = config.DB_CONFIG
        url = (
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            f"?charset={cfg['charset']}"
        )
        _engine = create_engine(url, echo=False)
        print(
            f"[connector] SQLAlchemy engine created -> "
            f"{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )
    return _engine


def get_session() -> Session:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory()


def reset_session_factory() -> None:
    global _session_factory
    _session_factory = None


def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
    print("[connector] SQLAlchemy engine disposed.")


def _extract_table_name(ddl: str) -> str:
    tokens = ddl.upper().split()
    try:
        idx = tokens.index("TABLE")
        offset = 1
        if idx + 1 < len(tokens) and tokens[idx + 1] == "IF":
            offset = 4
        return tokens[idx + offset].strip("`").lower()
    except (ValueError, IndexError):
        return "unknown"


def ping() -> bool:
    cfg = config.DB_CONFIG
    try:
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            charset=cfg["charset"],
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 AS ok;")
            row = cursor.fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        print(f"[connector] ping failed: {e}")
        return False


if __name__ == "__main__":
    print("connector smoke test")
    print("ping:", ping())
