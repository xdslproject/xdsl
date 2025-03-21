from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class BigIntegerType(ParametrizedAttribute, TypeAttribute):
    name = "bigint.bigint"


bigint = BigIntegerType()

BigInt = Dialect(
    "bigint",
    [],
    [
        BigIntegerType,
    ],
)
