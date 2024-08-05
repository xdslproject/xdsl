from enum import Enum, Flag
from typing import Any, Self


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


class StrFlag(Flag):

    @property
    def label(self) -> str:
        return self._label_

    @classmethod
    def from_label(cls, label: str) -> Self:
        for member in cls:
            if member.label == label:
                return member
        raise ValueError(f"No member with label '{label}' in {cls.__name__}")

    def __new__(cls, value: int, label: str | None = None) -> Self:

        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value: int, label: str | None = None) -> None:
        Flag.__init__(self)
        if label is None:
            if self.name is None:
                raise ValueError("Cannot infer label without a name")
            label = self.name.lower()
        self._label_ = label

    def __str__(self) -> str:
        return ",".join(f.label for f in self)
