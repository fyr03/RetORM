"""Global configuration for RetORM."""

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "123456",
    "database": "retorm",
    "charset": "utf8mb4",
}

NUM_SCHEMAS = 5
QUERIES_PER_SCHEMA = 20
Z3_TIMEOUT_SEC = 5
RANDOM_ROWS = 10
EXTRA_RANDOM_ROWS = 6
EDGE_ROWS = 4
ADVERSARIAL_ROWS = 4
STRESS_RETRY_BUDGET = 12

ENABLE_REF_PATH = True
ENABLE_CORE_PATH = False
ENABLE_TRUE_ORM_PATH = True

TRUE_ORM_JOIN_MODE = "relationship_preferred"
TRUE_ORM_MAX_IR_ATTEMPTS = 12
TRUE_ORM_ENTITY_PROJECTION = True
TRUE_ORM_LOADER_STRATEGY = "off"
TRUE_ORM_TOUCH_RELATIONSHIPS = False
