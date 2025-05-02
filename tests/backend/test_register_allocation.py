from collections.abc import Sequence

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_type import RegisterType
from xdsl.builder import Builder
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    var_operand_def,
    var_result_def,
)


@irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return {"x0": 0, "x1": 1, "a0": 0, "a1": 1}

    @classmethod
    def infinite_register_prefix(cls):
        return "y"


@irdl_op_definition
class TestAllocatableOp(IRDLOperation, RegisterAllocatableOperation):
    name = "test.allocatable"

    res = var_result_def()
    ops = var_operand_def()

    def __init__(
        self, operands: Sequence[SSAValue] = (), result_types: Sequence[Attribute] = ()
    ):
        super().__init__(operands=(operands,), result_types=(result_types,))


def op(operands: Sequence[SSAValue], *result_types: Attribute):
    return TestAllocatableOp(operands, result_types).results


def test_gather_allocated():
    u = TestRegister.unallocated()
    x0 = TestRegister.from_name("x0")
    x1 = TestRegister.from_name("x0")

    @Builder.implicit_region
    def no_preallocated_body() -> None:
        (v1,) = op((), u)
        (v2,) = op((), u)
        op((v1, v2), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(no_preallocated_body)
    )

    assert pa_regs == set()

    @Builder.implicit_region
    def one_preallocated_body() -> None:
        (v1,) = op((), u)
        (v2,) = op((), x0)
        op((v1, v2), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(one_preallocated_body)
    )

    assert pa_regs == {x0}

    @Builder.implicit_region
    def repeated_preallocated_body() -> None:
        (v1,) = op((), u)
        (v2,) = op((), x0)
        (v3,) = op((), x0)
        op((v1, v2, v3), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(repeated_preallocated_body)
    )

    assert pa_regs == {x0}

    @Builder.implicit_region
    def multiple_preallocated_body() -> None:
        (v1,) = op((), u)
        (v2,) = op((), x0)
        (v3,) = op((), x1)
        op((v1, v2, v3), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(multiple_preallocated_body)
    )

    assert pa_regs == {x0, x1}
