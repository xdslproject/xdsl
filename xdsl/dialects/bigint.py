"""Dialect for arbitrary-precision integers."""

from collections.abc import Callable
from typing import Any

from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class BigIntegerType(ParametrizedAttribute, TypeAttribute):
    """
    Type for arbitrary-precision integers (bigints), such as those in Python.
    """

    name = "bigint.bigint"

    @classmethod
    def to_xdsl(cls) -> Callable[[], Any]:
        return lambda: BigIntegerType()


bigint = BigIntegerType()

BigInt = Dialect(
    "bigint",
    [],
    [
        BigIntegerType,
    ],
)
