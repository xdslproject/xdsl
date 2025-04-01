"""Dialect for arbitrary-precision integers."""

from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class BigIntegerType(ParametrizedAttribute, TypeAttribute):
    """
    Type for arbitrary-precision integers (bigints), such as those in Python.
    """

    name = "bigint.bigint"


bigint = BigIntegerType()

BigInt = Dialect(
    "bigint",
    [],
    [
        BigIntegerType,
    ],
)
