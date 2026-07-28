# RetORM 技术文档

## 1. 项目目标

RetORM 是一个面向 ORM/SQL 翻译层的差分测试框架。它的核心思路是：

1. 先生成一个统一的中间表示 IR。
2. 再把同一个 IR 翻译成三条等价执行路径。
3. 分别在同一个 MySQL 上执行。
4. 通过结果比对发现翻译错误、组装错误、执行异常和支持缺失。

当前实现不使用 metamorphic testing，主要依赖“三路结果一致”作为 oracle。

## 2. 总体架构

项目由以下模块组成：

- `ir/nodes.py`：定义查询 IR 节点
- `generator/schema_gen.py`：随机生成数据库 schema
- `generator/ir_gen.py`：随机生成查询 IR
- `generator/data_gen.py`：生成测试数据并插入数据库
- `translators/python_ref.py`：Python 参考执行路径
- `translators/sql.py`：IR -> 原生 SQL
- `translators/sqlalchemy_orm.py`：IR -> SQLAlchemy Core
- `comparator/compare.py`：结果归一化与差分比较
- `runner.py`：主测试入口、调度、日志、bug 落盘
- `tests/manual_cases.py`：轻量回归测试

## 3. IR 设计

当前 IR 支持的节点包括：

- `Scan`
- `Join`（`INNER` / `LEFT`）
- `Filter`
- `GroupBy`
- `Having`
- `Project`

支持的条件/表达式包括：

- `Compare`
- `And`
- `Or`
- `Not`

支持的聚合函数包括：

- `SUM`
- `COUNT`
- `AVG`
- `MAX`
- `MIN`

## 4. Schema 生成

`generator/schema_gen.py` 负责随机生成表结构：

- 随机表名来自固定词表
- 每张表默认有 `id` 主键
- 其余列随机选择 `INT` / `FLOAT` / `VARCHAR(64)`
- 每列随机决定是否允许 `NULL`
- 表之间按概率生成外键关系

同时会生成：

- `CREATE TABLE` SQL
- `DROP TABLE` 顺序
- schema 打印信息

## 5. 查询生成

`generator/ir_gen.py` 从 schema 生成 IR，基本流程是：

1. 选一个主表作为 `Scan`
2. 按外键关系逐步扩展 `Join`
3. 可选添加 `Filter`
4. 可选添加 `GroupBy`
5. 可选添加 `Having`
6. 最外层添加 `Project`

当前实现还支持压力模式：

- `balanced`
- `join_heavy`
- `groupby_heavy`
- `duplicate_column_heavy`
- `null_heavy`

这些模式会倾向生成更危险的组合，例如：

- 多跳 join
- LEFT JOIN
- 右表列投影
- `NULL` 谓词
- 重复列名投影
- `COUNT(col)` / `AVG(col)` 等更容易出错的聚合组合

## 6. 数据生成

`generator/data_gen.py` 负责构造并插入测试数据。当前不是纯随机填充，而是分层生成：

- core rows：保证查询尽量非空
- edge rows：固定边界值，如 `0`、`-1`、`0.1`、`28.6`、`77.7`、空串、`NULL`
- adversarial rows：制造重复值、空值、外键偏斜、孤儿值等
- noise rows：补充更多随机扰动

此外，它还会根据 IR 提取 profile：

- 哪些列参与了过滤
- 哪些列出现在 LEFT JOIN 右侧
- 哪些列是可空列
- 哪些列更适合放边界值

Z3 可选开启：

- 如果 IR 约束可解，会先尝试构造满足约束的数据
- 不可解或超时则回退到随机/分层生成

## 7. 三条执行路径

### 7.1 Python 参考路径

`translators/python_ref.py` 直接在 Python 中模拟 SQL 语义：

- `Scan`：从数据库读取原始表
- `Join`：执行内连接/左连接
- `Filter`：按三值逻辑过滤
- `GroupBy`：按分组键聚合
- `Having`：对聚合结果再过滤
- `Project`：返回指定列

它是差分测试的参考基准。

### 7.2 原生 SQL 路径

`translators/sql.py` 把 IR 翻译成原生 SQL：

- 生成 `SELECT / FROM / WHERE / GROUP BY / HAVING`
- 支持 `INNER JOIN` 和 `LEFT JOIN`
- 支持 `IS NULL` / `IS NOT NULL`
- 支持聚合别名引用

### 7.3 SQLAlchemy Core 路径

`translators/sqlalchemy_orm.py` 通过 SQLAlchemy Core 组装查询：

- 递归收集上下文
- 组装 `select()`
- 组装 `join() / outerjoin()`
- 组装 `where() / group_by() / having()`
- 最终由 SQLAlchemy 发送到同一个 MySQL 执行

注意：当前实现底层是 SQLAlchemy Core，而不是传统 ORM Model 映射。

## 8. 结果比较

`comparator/compare.py` 负责结果归一化和比较，重点处理：

- 列名格式差异
- ORM label 差异
- 重复列名
- `Decimal` / `float` / `int` 差异
- `NaN` / `None`
- 无 `ORDER BY` 时的行顺序差异
- 浮点近似误差

目前无序结果比较采用“语义等价的多重集合匹配”，避免因为浮点排序不稳定而误报。

## 9. 测试入口

`runner.py` 负责串联整个流程：

1. 生成 schema
2. 建表
3. 生成 IR
4. 生成并插入数据
5. 执行三条路径
6. 比较结果
7. 统计覆盖率和 bug
8. 生成详细日志和复现脚本

常用参数：

- `--schemas`：schema 轮数
- `--queries`：每个 schema 下的查询数
- `--tables`：表数量
- `--cols`：每表列数
- `--rows`：基础插入行数
- `--seed`：固定随机种子
- `--no-z3`：关闭 Z3
- `--verbose`：打印详细过程

## 10. 测试范围

当前测试主要覆盖：

- 单表查询
- 多表 join
- LEFT JOIN
- WHERE 过滤
- GROUP BY
- HAVING
- 重复列名投影
- `NULL` 谓词
- `COUNT(*)`
- `COUNT(col)`
- `AVG / SUM / MAX / MIN`
- 复杂布尔条件 `AND / OR / NOT`

当前 oracle 规则：

- 三路结果不一致 => bug
- 执行异常 => bug
- 原生 SQL 可表达但 SQLAlchemy Core 路径不支持 => bug

## 11. 日志与产物

运行后会生成：

- `logs_detail/<ts>.log`：详细过程日志
- `logs/<ts>.log`：运行统计日志
- `logs_bug/<bug_ts>.log`：单个 bug 的完整分析
- `bugs/<run_ts>/bug_N.py`：可直接复现的脚本

## 12. 现有限制

- 目前不做 metamorphic testing
- 目前不支持 `ORDER BY`
- 目前仍以 MySQL 为唯一后端
- 当前重点仍在翻译层和组装层，不在复杂执行优化

## 13. 当前结论

这套实现已经能够：

- 生成 schema
- 生成对抗性数据
- 生成复杂 IR
- 翻译成三条等价路径
- 在同一 MySQL 上执行
- 自动比对结果并落盘 bug

也就是说，它已经具备了一个可用的 RetORM 差分测试原型框架。
