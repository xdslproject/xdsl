from __future__ import annotations

from typing import Any, NoReturn

from typing_extensions import Self


class Descriptor:
    def __init__(self, name: str) -> None:
        self.name = name

    def __set__(self, instance: Self, value: Any):
        instance.__dict__[self.name] = value

    def __delete__(self, instance: Self) -> NoReturn:
        raise AttributeError(f"Cannot delete {self.name}")


class Typed(Descriptor):
    ty: type

    def __set__(self, instance: Self, value: Any):
        if not isinstance(value, self.ty):
            raise TypeError(f"Expected {self.ty}")
        super().__set__(instance, value)


class Integer(Typed):
    ty = int


class Float(Typed):
    ty = float


class String(Typed):
    ty = str


class Positive(Descriptor):
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError("Expected >= 0")
        super().__set__(instance, value)


class PosInteger(Integer, Positive):
    pass


class PosFloat(Float, Positive):
    pass


class Sized(Descriptor):
    def __init__(self, *args, maxlen, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if len(value) > self.maxlen:
            raise ValueError("Too big")
        super().__set__(instance, value)
