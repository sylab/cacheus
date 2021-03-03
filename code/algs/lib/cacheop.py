from enum import Enum


class CacheOp(Enum):
    HIT = 0
    INSERT = 1
    FILTER = 2
