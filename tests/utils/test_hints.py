import re
from typing import Generic

import pytest
from typing_extensions import TypeVar

from xdsl.utils.hints import get_type_var_mapping

T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


class A(Generic[T1, T2]): ...


class B(A[T1, int], Generic[T1]): ...


class C(B[str]): ...


class D(A[T0, T0], Generic[T0]): ...


class D2(A[T0, T0]): ...


class E(Generic[T2]): ...


class F(C, E[str]): ...


def test_get_type_var_mapping():
    assert get_type_var_mapping(A) == ((T1, T2), {})
    assert get_type_var_mapping(B) == ((T1,), {T2: int})
    assert get_type_var_mapping(C) == (
        (),
        {
            T1: str,
            T2: int,
        },
    )
    assert get_type_var_mapping(D) == (
        (T0,),
        {
            T1: T0,
            T2: T0,
        },
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid definition D2, generic classes must subclass `Generic`."
        ),
    ):
        assert get_type_var_mapping(D2) == (
            (T0,),
            {
                T1: T0,
                T2: T0,
            },
        )
    assert get_type_var_mapping(E) == ((T2,), {})
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Error extracting assignments of TypeVars of F, inconsistent assignments to ~T2 in superclasses: str, int."
        ),
    ):
        assert get_type_var_mapping(F) == ()
