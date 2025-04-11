from xdsl.dialects.test import TestOp
from xdsl.ir import AttributeCovT, OpResult


def TestSSAValue(t: AttributeCovT) -> OpResult[AttributeCovT]:
    op = TestOp(result_types=(t,))
    return op.results[0]  # pyright: ignore[reportReturnType]
