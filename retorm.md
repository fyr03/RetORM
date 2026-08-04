# RetORM 技术文档

## 1. 项目目标

RetORM 是一个面向 SQL 翻译层与 ORM 翻译层的一致性差分测试框架。  
当前版本的核心目标不是做数据库性能测试，也不是做通用 SQL Fuzzing，而是尽量系统地回答一个问题：

- 同一个查询语义，`raw SQL` 与 `true ORM` 路径是否会得到一致结果。

当前主测试思路是：

1. 随机生成 `Schema`
2. 在该 `Schema` 上生成随机但结构受控的 `IR`
3. 基于 `IR` 生成测试数据并插入 MySQL
4. 将同一份 `IR` 分别交给多条执行路径
5. 比较结果与 ORM 事实信息
6. 将执行异常、结果不一致、路径支持缺失等情况记录为问题

与较早版本不同，当前项目的主比较重心已经从“SQL vs Core-like ORM API”转向：

- `raw SQL`
- `SQLAlchemy true ORM`

`python_ref` 仍然保留，但主要作为语义参考与诊断辅助，不再是唯一主裁判。

---

## 2. 当前总体架构

当前项目主要由以下模块组成：

- `ir/nodes.py`
  - 定义查询中间表示 IR
- `generator/schema_gen.py`
  - 随机生成数据库 schema
- `generator/ir_gen.py`
  - 随机生成查询 IR
- `generator/data_gen.py`
  - 生成并插入随机数据、边界数据、对抗性数据
- `translators/python_ref.py`
  - Python 参考执行路径
- `translators/sql.py`
  - `IR -> raw SQL`
- `translators/sqlalchemy_true_orm.py`
  - `IR -> SQLAlchemy true ORM`
- `comparator/compare.py`
  - 结果归一化与差分比较
- `db/connector.py`
  - MySQL 连接、SQL 执行、Engine / Session 管理
- `runner.py`
  - 主调度入口、覆盖率统计、日志、bug 落盘

此外，仓库里仍保留旧的 `translators/sqlalchemy_orm.py`，但当前默认主路径已经不是它。默认配置见 [config.py](/d:/Project/RetORM/config.py)：

- `ENABLE_TRUE_ORM_PATH = True`
- `ENABLE_CORE_PATH = False`

---

## 3. IR 设计

### 3.1 关系节点

当前 `IR` 已支持的主要关系节点包括：

- `Scan`
- `DerivedTable`
- `Filter`
- `Join`
- `GroupBy`
- `Having`
- `Project`
- `Distinct`
- `OrderBy`
- `LimitOffset`
- `SetQuery`

其中 `Join` 支持：

- `INNER`
- `LEFT`

`SetQuery` 支持：

- `UNION`
- `INTERSECT`
- `EXCEPT`

### 3.2 条件与表达式

当前 `IR` 支持的条件节点包括：

- `Compare`
- `InList`
- `Between`
- `Like`
- `Exists`
- `InSubquery`
- `And`
- `Or`
- `Not`

当前 `IR` 支持的值表达式包括：

- 列引用
- 常量
- `ArithExpr`
- `CaseWhen`
- `ScalarSubquery`
- `WindowExpr`

### 3.3 聚合与窗口

聚合函数支持：

- `SUM`
- `COUNT`
- `AVG`
- `MAX`
- `MIN`

窗口函数支持：

- `ROW_NUMBER`
- `RANK`
- `DENSE_RANK`
- `SUM`
- `COUNT`
- `AVG`
- `MAX`
- `MIN`

### 3.4 IR 的作用

整个框架的关键在于：  
所有路径都消费同一份 IR，而不是各自独立随机生成 SQL 或 ORM 查询。这样能把差分测试的焦点集中在“翻译是否一致”上。

---

## 4. Schema 生成

[generator/schema_gen.py](/d:/Project/RetORM/generator/schema_gen.py) 负责随机生成 schema。其设计目标不是完全任意，而是“随机 + 易出错结构优先”。

当前 schema 生成的主要特征：

- 表名来自固定词表，如 `orders`、`users`、`items`、`reviews`
- 每张表默认包含 `id` 主键
- 普通列类型从以下集合中随机选择
  - `INT`
  - `FLOAT`
  - `VARCHAR(64)`
- 列是否可空按概率生成
- 外键按概率生成

当前版本还额外生成若干更容易暴露 ORM / SQL 差异的 schema 形状：

- 自引用外键 `self FK`
- 同一张表对同一目标表的多条外键
- 类 association table 结构
- hub-like 结构
- backlink / extra FK

这些控制由 [config.py](/d:/Project/RetORM/config.py) 中的参数决定，例如：

- `SCHEMA_SELF_FK_PROB`
- `SCHEMA_MULTI_FK_SAME_TARGET_PROB`
- `SCHEMA_ASSOC_TABLE_PROB`
- `SCHEMA_HUB_TABLE_PROB`

Schema 生成模块同时提供：

- `CREATE TABLE` SQL
- `DROP TABLE` SQL
- Python 级 `Schema` 元数据

---

## 5. 查询生成

[generator/ir_gen.py](/d:/Project/RetORM/generator/ir_gen.py) 负责随机生成 IR。当前实现已经不再只是“简单拼接 Scan / Join / Filter”，而是会根据 stress mode 定向生成更危险的查询结构。

### 5.1 基本生成流程

一个查询通常会经历如下构造流程：

1. 选择起始表，生成 `Scan`
2. 根据外键关系扩展 `Join`
3. 按概率加入 `Filter`
4. 按概率加入 `GroupBy`
5. 按概率加入 `Having`
6. 生成 `Project`
7. 进一步包装为
   - `Distinct`
   - `OrderBy`
   - `LimitOffset`
   - `DerivedTable`
   - `SetQuery`

### 5.2 当前重点 stress modes

当前生成器已经包含一批结构导向与 ORM 导向的模式，例如：

- `balanced`
- `join_heavy`
- `groupby_heavy`
- `duplicate_column_heavy`
- `null_heavy`
- `distinct_heavy`
- `orderby_heavy`
- `subquery_heavy`
- `derived_heavy`
- `window_heavy`
- `setop_heavy`
- `self_join_heavy`
- `entity_heavy`
- `entity_dedup_heavy`
- `distinct_entity_heavy`
- `limit_joined_entity_heavy`
- `relationship_heavy`
- `relationship_orderby_heavy`
- `loader_heavy`
- `loader_strategy_heavy`
- `combo_heavy`
- `orm_combo_heavy`

### 5.3 生成器当前重点覆盖的风险面

当前生成器重点打击以下几类容易出错的组合：

- `LEFT JOIN + NULL`
- `LEFT JOIN + right-side projection`
- `LEFT JOIN + predicate`
- `entity projection + scalar projection`
- `relationship-heavy join chain`
- `self join + alias`
- `DISTINCT + ORDER BY + LIMIT`
- `GroupBy + Having + aggregate alias`
- `DerivedTable + outer projection`
- `SetQuery + complex branch`
- `WindowExpr + outer ORDER BY`

### 5.4 outcome-guided generation

当前 `runner.py` 会根据运行中观察到的覆盖率与结果分布，动态调节后续 stress mode 权重，而不是始终均匀随机。这一层反馈主要基于：

- 结构覆盖缺口
- true ORM API 覆盖缺口
- 某些模式的空结果率
- 某些模式的行预算是否需要提升

因此现在的查询生成已经不是静态随机，而是带有一定“结果导向”调整能力。

---

## 6. 数据生成

[generator/data_gen.py](/d:/Project/RetORM/generator/data_gen.py) 的设计目标是：  
不要只靠“多跑查询”找 bug，也要靠“更刁钻的数据”找 bug。

### 6.1 数据分层

当前数据生成大致分为几层：

- `RANDOM_ROWS`
  - 基础随机行
- `EXTRA_RANDOM_ROWS`
  - 额外随机扰动
- `EDGE_ROWS`
  - 固定边界值
- `ADVERSARIAL_ROWS`
  - 对抗性构造

### 6.2 当前重点注入的数据特征

包括但不限于：

- `0`
- `1`
- `-1`
- `NULL`
- 空字符串
- 重复值
- 浮点边界值
- 连接链可命中 / 不可命中混合
- 外键断裂
- 只在右表出现的值
- 可能触发聚合 / 去重 / 空扩展差异的数据分布

### 6.3 anchor row 与 selectivity 调节

为了避免“查询很多，但都没有有效结果”的问题，当前版本加入了：

- anchor row reinforcement
  - 强化某些可连接、可命中谓词的行
- 根据 stress mode 提升行预算
  - 例如对 `derived_heavy`、`window_heavy`、子查询相关模式适当加大数据量
- 根据运行时空结果率做 row budget feedback

### 6.4 Z3

当前仍支持可选的 Z3 辅助数据生成：

- 若约束可解，则尝试构造满足条件的数据
- 若超时或不可解，则回退到随机 / 分层生成

CLI 中可使用：

- `--no-z3`

关闭这一路径。

---

## 7. 执行路径

## 7.1 Python 参考路径

[translators/python_ref.py](/d:/Project/RetORM/translators/python_ref.py) 在 Python 中直接解释 IR 语义。

它的价值主要在于：

- 提供一个不依赖 SQLAlchemy 的参考执行结果
- 帮助区分“SQL 与 true ORM 同时偏离”还是“只有某一路径偏离”

当前它仍然重要，但在 bug 分类里更偏向辅助诊断，而不是唯一主裁判。

## 7.2 raw SQL 路径

[translators/sql.py](/d:/Project/RetORM/translators/sql.py) 将 IR 翻译为原生 SQL，并通过 [db/connector.py](/d:/Project/RetORM/db/connector.py) 发到 MySQL 执行。

当前 SQL 路径已覆盖：

- `SELECT / FROM / WHERE`
- `INNER JOIN / LEFT JOIN`
- `GROUP BY / HAVING`
- `DISTINCT`
- `ORDER BY`
- `LIMIT / OFFSET`
- `CASE WHEN`
- 算术表达式
- `IN / EXISTS / BETWEEN / LIKE`
- `DerivedTable`
- `ScalarSubquery`
- `UNION / INTERSECT / EXCEPT`
- `WindowExpr`

## 7.3 true ORM 路径

[translators/sqlalchemy_true_orm.py](/d:/Project/RetORM/translators/sqlalchemy_true_orm.py) 是当前项目最核心的路径。

它与旧的 Core-like 路径不同，当前实现确实使用了 SQLAlchemy ORM 层能力：

- 动态生成 declarative mapped classes
- 基于 FK 元数据生成 `relationship()`
- 使用 `Session`
- 使用 `aliased()` 处理别名与自连接
- 支持 relationship-derived join
- 支持 explicit join fallback
- 支持 entity projection
- 支持 entity + scalar 混合投影
- 支持 loader strategy
  - `joinedload`
  - `selectinload`
- 支持 derived table / scalar subquery / set op / window

### 7.3.1 true ORM 的输出

true ORM 路径当前返回的是一个 `TrueORMResult`，其中包括：

- `rows`
  - 用于与 SQL 路径直接比较的扁平化结果
- `facts`
  - ORM 语义相关的附加事实
- `compiled_sql`
  - 按采样率记录的 SQLAlchemy 编译后 SQL

### 7.3.2 当前采集的 ORM facts

`TrueORMFacts` 当前包含：

- `entity_tables`
- `entity_pk_columns`
- `entity_pks`
- `duplicate_entity_pks`
- `loaded_relationships`
- `expected_loaded_relationships`
- `identity_map_size`
- `materialized_entity_count`

这些 facts 的作用不是替代 row comparison，而是补充 row comparison，看是否发生了：


- 实体 materialization 异常
- identity map 表现异常
- relationship loader 行为异常
- 重复 entity 行处理异常

### 7.3.3 支持检查

当前 true ORM 路径在执行前会调用 `supports_true_orm(ir)` 做静态支持检查。  
不支持的形状会被统计为：

- `true_orm unsupported`

并跳过主 bug 统计，避免“路径尚未实现”污染真正的一致性 bug 结果。

---

## 8. 比较器与 bug 分类

[comparator/compare.py](/d:/Project/RetORM/comparator/compare.py) 负责结果归一化与比较。

### 8.1 当前归一化处理

比较器会处理以下常见噪声：

- 列名前缀差异
- ORM / SQL label 格式差异
- 重复列名重编号
- `Decimal` 与 `float`
- `NaN` 与 `None`
- 无 `ORDER BY` 时的无序比较
- 小范围浮点误差

### 8.2 strict compare

当前比较器支持 strict mode：

- CLI 参数：`--strict-compare`

strict mode 下会更少做宽松归一化，并强制按顺序比较，适合排查：

- label 归一化问题
- 顺序相关问题
- 注入故障是否真的能被抓住

### 8.3 当前 bug 分类

`runner.py` 当前会把问题大致分成：

- `Execution Error`
  - 执行异常
- `SQL vs True ORM Divergence`
  - SQL 结果与 true ORM 结果不同
- `True ORM Fact Mismatch`
  - 行结果相同，但 ORM facts 不一致
- `Ref Path Anomaly`
  - 参考路径异常，仅用于诊断

其中真正主 bug 关注的是：

- 执行异常
- `SQL vs true ORM`
- `true ORM facts`

---

## 9. Runner 工作流

[runner.py](/d:/Project/RetORM/runner.py) 串联整个测试过程。

单条查询的大致流程如下：

1. 生成 schema
2. 建表
3. 生成 IR
4. 生成并插入数据
5. 执行 `python_ref`
6. 执行 `raw SQL`
7. 执行 `true ORM`
8. 记录 true ORM coverage
9. 比较结果与 facts
10. 统计覆盖率
11. 对异常或差异生成 bug report 与复现脚本
12. 按采样率执行 fault smoke

### 9.1 当前 CLI 参数

当前主要参数包括：

- `--schemas`
- `--queries`
- `--tables`
- `--cols`
- `--rows`
- `--seed`
- `--fault-smoke-rate`
- `--strict-compare`
- `--true-orm-fault`
- `--no-z3`
- `--verbose`

### 9.2 示例

基础运行：

```bash
python runner.py --schemas 5 --queries 50 --tables 3 --cols 4
```

更严格比较：

```bash
python runner.py --schemas 5 --queries 50 --tables 3 --cols 4 --strict-compare
```

观察 fault smoke：

```bash
python runner.py --schemas 1 --queries 100 --fault-smoke-rate 1.0 --verbose
```

---

## 10. 覆盖率统计

当前版本的 runner 不只统计“跑了多少条查询”，还会统计多种覆盖信息。

### 10.1 结构覆盖

包括：

- single table
- join
- multi join
- self join
- left join
- filter
- group by
- having
- distinct
- order by
- limit/offset
- set query
- entity projection
- entity + scalar
- duplicate projection
- null predicate

### 10.2 语法覆盖

包括：

- `IN`
- `BETWEEN`
- `LIKE`
- arithmetic
- `CASE WHEN`
- window
- derived table
- subquery
- `EXISTS`
- `IN-subquery`
- `DISTINCT + ORDER + LIMIT`

### 10.3 结果覆盖

包括：

- single-row result
- multi-row result
- duplicate-row result
- null-containing result
- entity-duplicate result
- left-join null-extension result
- aggregation-null result

### 10.4 true ORM API 覆盖

包括：

- relationship join
- explicit join
- entity materialization
- entity + scalar
- joinedload
- selectinload
- relationship touch
- self alias
- set query
- scalar subquery
- window expression
- derived table
- `IN` limit-wrap

---

## 11. fault smoke 机制

当前版本额外加入了 `fault smoke`，其目标不是找真实 bug，而是验证测试框架自己是否“有能力抓住故意注入的错”。

### 11.1 工作方式

在主查询正常比较完成后，runner 会按采样率对部分查询额外再跑一次“带故障的 true ORM”。

当前候选故障模式包括：

- `inner_for_left_join`
- `reverse_order`
- `drop_offset`
- `count_star`
- `null_eq_false`

### 11.2 统计含义

运行日志中的：

- `attempts`
- `detected`
- `missed`

表示：

- 尝试注入了多少次故障
- 注入后差异是否被框架抓住
- 注入后差异没有暴露出来

注意：

- `fault smoke` 不是正式 bug
- 它默认不进入主 bug 计数
- 它是框架自检，不是产品缺陷统计

---

## 12. 输出产物

运行后会生成以下输出：

- `logs_detail/<ts>.log`
  - 详细运行日志
- `logs/<ts>.log`
  - 汇总统计日志
- `logs_bug/<bug_ts>.log`
  - 单个问题的详细分析日志
- `bugs/<run_ts>/bug_N.py`
  - 可直接执行的复现脚本

bug 详情中通常会包含：

- schema SQL
- IR
- 插入数据
- raw SQL
- true ORM 伪代码
- 三路结果
- true ORM facts

---

## 13. 当前默认测试重心

截至当前代码版本，RetORM 的主测试重心可以概括为：

- 扩展 SQL / ORM 共有语义面的测试范围
- 扩展复杂查询结构
- 扩展 true ORM 特有 API 覆盖
- 尽量降低误报，但不靠“缩小查询范围”来降低误报

默认主比较重点是：

- `raw SQL vs true ORM`

而不是旧的 SQLAlchemy Core-like 路径。

---

## 14. 当前局限

虽然当前版本已经比早期实现完整很多，但仍有一些局限需要明确：

- `tests/manual_cases.py` 当前已删除，暂时没有系统性的手工回归集
- 一些窗口函数场景如果 `ORDER BY` 不能唯一确定顺序，会天然存在非确定性
- `fault smoke` 是抽样且概率性的，不是完备自检
- true ORM 当前仍然会把结果扁平化成 rows 再做主比较，因此对象语义的深层行为主要通过 `facts` 辅助观察
- 某些极端 IR 形状如果 true ORM 路径无法表达，会被记为 unsupported 而不是直接当作 bug

---

## 15. 当前实现结论

从代码实现上看，当前 RetORM 已经具备一个较完整的 differential testing 原型，能够：

- 生成带复杂结构的随机 schema
- 生成随机、边界、对抗性测试数据
- 生成复杂 IR
- 同时覆盖 SQL 共有语义与大量 true ORM 相关路径
- 执行 `python_ref`、`raw SQL`、`true ORM`
- 比较结果与 ORM facts
- 统计覆盖率
- 输出详细日志与复现脚本
- 通过 fault smoke 验证测试框架自身的抓错能力

因此，当前版本已经不只是一个“随机 SQL 生成器”，而是一个围绕 `SQL vs true ORM` 一致性问题构建的、带覆盖反馈和自检能力的测试框架。
