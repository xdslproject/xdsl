from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    """
    Homemade StrEnum. StrEnum is standard in Python>=3.11.
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ):
        return name.lower()

    def __str__(self) -> str:
        return self.value
