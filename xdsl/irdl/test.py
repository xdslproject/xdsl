from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeGuard, TypeVar


class A: ...


@dataclass
class B(A):
    b: int


class C(A): ...


class D(A): ...


class VerifyException(Exception): ...


CovT = TypeVar("CovT", bound=A, covariant=True)
T = TypeVar("T", bound=A)


class Constraint(Generic[CovT]):

    def match(self, a: A) -> TypeGuard[CovT]:
        try:
            self.verify(a)
            return True
        except VerifyException:
            return False

    def verify(self, a: A) -> None:
        raise NotImplementedError

    def __or__(self, value: Constraint[T], /) -> AnyOf[CovT | T]:
        return AnyOf((self, value))


@dataclass
class Eq(Generic[CovT], Constraint[CovT]):

    attr: CovT

    def verify(self, a: A) -> None:
        if a != self.attr:
            raise VerifyException()


@dataclass
class Is(Generic[CovT], Constraint[CovT]):

    attr: type[CovT]

    def verify(self, a: A) -> None:
        if not isinstance(a, self.attr):
            raise VerifyException()


@dataclass
class AnyOf(Generic[CovT], Constraint[CovT]):
    bla: tuple[Constraint[A], ...]

    def verify(self, a: A) -> None:
        for constr in self.bla:
            try:
                constr.verify(a)
                return
            except VerifyException:
                continue
        raise VerifyException()

    def __or__(self, value: Constraint[T], /) -> AnyOf[CovT | T]:
        return AnyOf((*self.bla, value))


hello = Eq(B(1)) | Is(C) | Is(D)
