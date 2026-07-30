"""
db/connector.py

封装所有和 MySQL 直接交互的操作。
三条翻译路径（python_ref / sql / sqlalchemy_orm）都通过这里拿连接，
不各自管理连接逻辑。

主要职责：
  1. 建立 / 关闭 MySQL 连接
  2. 执行建表 SQL（schema 初始化）
  3. 执行 INSERT（测试数据写入）
  4. 执行原生 SELECT SQL，返回统一格式的结果
  5. 清理：DROP TABLE / 清空数据
  6. 提供 SQLAlchemy engine（供 sqlalchemy_orm.py 使用）
"""

import sys
import pymysql
import pymysql.cursors
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional

# 把项目根目录加入路径，确保能 import config
sys.path.insert(0, __file__.rsplit("/db", 1)[0])
import config


# ---------------------------------------------------------------------------
# 底层连接：pymysql（原生 SQL 路径 和 程序逻辑路径 使用）
# ---------------------------------------------------------------------------

def get_connection() -> pymysql.connections.Connection:
    """
    建立并返回一个 pymysql 连接。
    调用方用完后需要自己 close()，或者用 with 语句。
    
    用法：
        conn = get_connection()
        ...
        conn.close()
    
    或：
        with get_connection() as conn:   # 注意：pymysql 的 with 管理的是事务，不是连接
            ...
    """
    cfg = config.DB_CONFIG
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg["charset"],
        cursorclass=pymysql.cursors.DictCursor,  # 返回 dict 而不是 tuple，方便后续处理
        autocommit=True,
    )
    return conn


def execute_sql(sql: str, params=None) -> List[Dict[str, Any]]:
    """
    执行一条原生 SQL，返回所有行。
    每次调用都新建连接、执行、关闭，适合测试场景（不需要连接复用）。
    
    返回值：
        SELECT 语句 → List[Dict]，每个 dict 是一行，key 是列名
        非 SELECT 语句（CREATE / INSERT / DROP）→ 空列表
    
    用法：
        rows = execute_sql("SELECT * FROM orders WHERE amount > %s", (100,))
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            # DictCursor 对非 SELECT 语句 fetchall() 返回空列表
            result = cursor.fetchall()
            return list(result)
    finally:
        conn.close()


def execute_many(sql: str, data: List[tuple]) -> None:
    """
    批量执行 INSERT，data 是参数元组的列表。
    
    用法：
        execute_many(
            "INSERT INTO orders (user_id, amount) VALUES (%s, %s)",
            [(1, 50.0), (2, 200.0)]
        )
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(sql, data)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema 管理
# ---------------------------------------------------------------------------

def init_database() -> None:
    """
    确保 config 里指定的 database 存在。
    用 root 连接时先不指定 database，创建后再切换。
    如果已存在则跳过。
    """
    cfg = config.DB_CONFIG
    # 先不带 database 连接
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
    """
    按顺序执行一组 CREATE TABLE 语句。
    ddl_statements 是字符串列表，每个元素是一条完整的 CREATE TABLE SQL。
    
    用法：
        create_tables([
            "CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, age INT)",
            "CREATE TABLE IF NOT EXISTS orders (id INT PRIMARY KEY, user_id INT, amount FLOAT)",
        ])
    """
    for ddl in ddl_statements:
        execute_sql(ddl)
        # 从 DDL 里提取表名用于日志
        table_name = _extract_table_name(ddl)
        print(f"[connector] table `{table_name}` created.")


def drop_tables(table_names: List[str]) -> None:
    """
    按顺序 DROP 指定的表（IF EXISTS）。
    传入的顺序应该考虑外键依赖，先 drop 子表再 drop 父表。
    
    用法：
        drop_tables(["orders", "users"])
    """
    for name in table_names:
        execute_sql(f"DROP TABLE IF EXISTS `{name}`;")
        print(f"[connector] table `{name}` dropped.")


def truncate_tables(table_names: List[str]) -> None:
    """
    清空指定表的数据，保留表结构。
    用于在同一个 schema 下跑多组测试数据时重置状态。
    """
    for name in table_names:
        execute_sql(f"TRUNCATE TABLE `{name}`;")


def insert_rows(table: str, columns: List[str], rows: List[tuple]) -> None:
    """
    向指定表批量插入数据。
    
    参数：
        table:   表名
        columns: 列名列表，顺序和 rows 里的元组对应
        rows:    数据行列表，每行是一个元组
    
    用法：
        insert_rows("orders", ["id", "user_id", "amount"], [
            (1, 1, 50.0),
            (2, 1, 80.0),
            (3, 2, 200.0),
        ])
    """
    if not rows:
        return
    col_str         = ", ".join(f"`{c}`" for c in columns)
    placeholder_str = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO `{table}` ({col_str}) VALUES ({placeholder_str});"
    execute_many(sql, rows)
    print(f"[connector] inserted {len(rows)} rows into `{table}`.")


# ---------------------------------------------------------------------------
# SQLAlchemy engine（供 sqlalchemy_orm.py 使用）
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None  # 模块级单例，避免重复创建


def get_engine() -> Engine:
    """
    返回 SQLAlchemy engine 单例。
    连接串格式：mysql+pymysql://user:password@host:port/database
    
    用法（在 sqlalchemy_orm.py 里）：
        from db.connector import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
    """
    global _engine
    if _engine is None:
        cfg = config.DB_CONFIG
        url = (
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            f"?charset={cfg['charset']}"
        )
        _engine = create_engine(url, echo=False)
        print(f"[connector] SQLAlchemy engine created → {cfg['host']}:{cfg['port']}/{cfg['database']}")
    return _engine


def dispose_engine() -> None:
    """关闭 SQLAlchemy engine 的连接池，测试结束时调用。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
        print("[connector] SQLAlchemy engine disposed.")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _extract_table_name(ddl: str) -> str:
    """从 CREATE TABLE 语句里提取表名，仅用于日志输出。"""
    tokens = ddl.upper().split()
    try:
        idx = tokens.index("TABLE")
        # 跳过可能的 IF NOT EXISTS（三个词）
        offset = 1
        if tokens[idx + 1] == "IF":
            offset = 4   # TABLE IF NOT EXISTS <name>
        name_token = tokens[idx + offset]
        return name_token.strip("`").lower()
    except (ValueError, IndexError):
        return "unknown"


def ping() -> bool:
    """
    测试 MySQL 连接是否正常，返回 True 表示连通。
    注意：不依赖 database 是否存在，直接连 MySQL 服务本身。
    """
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


# ---------------------------------------------------------------------------
# 直接运行此文件时：做连接测试 + 建库 + 简单建表 / 插数据 / 查询 / 清理
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=== Step 1: ping（不依赖 database 存在）===")
    ok = ping()
    print(f"连接状态: {'✓ 正常' if ok else '✗ 失败'}")
    if not ok:
        sys.exit(1)

    print("\n=== Step 2: init database ===")
    init_database()  # 建库放在 ping 之后，顺序正确

    print("\n=== Step 3: create tables ===")
    create_tables([
        """
        CREATE TABLE IF NOT EXISTS `users` (
            `id`  INT PRIMARY KEY,
            `age` INT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS `orders` (
            `id`      INT PRIMARY KEY,
            `user_id` INT NOT NULL,
            `amount`  FLOAT NOT NULL
        );
        """,
    ])

    print("\n=== Step 4: insert rows ===")
    insert_rows("users",  ["id", "age"],             [(1, 25), (2, 17), (3, 30)])
    insert_rows("orders", ["id", "user_id", "amount"],
                [(1, 1, 50.0), (2, 1, 80.0), (3, 2, 200.0), (4, 3, 30.0)])

    print("\n=== Step 5: query ===")
    rows = execute_sql("SELECT * FROM `orders` WHERE amount > %s;", (60,))
    print("orders WHERE amount > 60:")
    for r in rows:
        print(" ", r)

    print("\n=== Step 6: SQLAlchemy engine ===")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) AS cnt FROM `orders`;"))
        print("orders count via SQLAlchemy:", result.fetchone()._mapping["cnt"])

    print("\n=== Step 7: cleanup ===")
    drop_tables(["orders", "users"])
    dispose_engine()

    print("\n全部通过 ✓")
