from enum import Enum


class DCEMode(Enum):
    NONE = 0
    SIMPLE = 1
    RECURSIVE = 2


dce_mode = DCEMode.RECURSIVE