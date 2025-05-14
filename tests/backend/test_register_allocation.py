from collections.abc import Sequence

from xdsl.backend.register_allocatable import (
    HasRegisterConstraints,
    RegisterAllocatableOperation,
    RegisterConstraints,
)
from xdsl.backend.register_type import RegisterType
from xdsl.builder import Builder
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
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
class TestAllocatableOp(IRDLOperation, HasRegisterConstraints):
    name = "test.allocatable"

    in_operands = var_operand_def()
    inout_operands = var_operand_def()
    out_results = var_result_def()
    inout_results = var_result_def()

    irdl_options = [AttrSizedOperandSegments(), AttrSizedResultSegments()]

    def __init__(
        self,
        in_operands: Sequence[SSAValue],
        inout_operands: Sequence[SSAValue],
        out_result_types: Sequence[Attribute],
        inout_result_types: Sequence[Attribute],
    ):
        super().__init__(
            operands=(in_operands, inout_operands),
            result_types=(out_result_types, inout_result_types),
        )

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            self.in_operands,
            self.out_results,
            tuple(zip(self.inout_operands, self.inout_results)),
        )


def op(
    ins: Sequence[SSAValue],
    *out_result_types: Attribute,
    inouts: Sequence[SSAValue] = (),
):
    return TestAllocatableOp(
        ins, inouts, out_result_types, tuple(val.type for val in inouts)
    )


def test_gather_allocated():
    u = TestRegister.unallocated()
    x0 = TestRegister.from_name("x0")
    x1 = TestRegister.from_name("x0")

    @Builder.implicit_region
    def no_preallocated_body() -> None:
        (v1,) = op((), u).results
        (v2,) = op((), u).results
        op((v1, v2), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(no_preallocated_body)
    )

    assert pa_regs == set()

    @Builder.implicit_region
    def one_preallocated_body() -> None:
        (v1,) = op((), u).results
        (v2,) = op((), x0).results
        op((v1, v2), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(one_preallocated_body)
    )

    assert pa_regs == {x0}

    @Builder.implicit_region
    def repeated_preallocated_body() -> None:
        (v1,) = op((), u).results
        (v2,) = op((), x0).results
        (v3,) = op((), x0).results
        op((v1, v2, v3), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(repeated_preallocated_body)
    )

    assert pa_regs == {x0}

    @Builder.implicit_region
    def multiple_preallocated_body() -> None:
        (v1,) = op((), u).results
        (v2,) = op((), x0).results
        (v3,) = op((), x1).results
        op((v1, v2, v3), u)

    pa_regs = set(
        RegisterAllocatableOperation.iter_all_used_registers(multiple_preallocated_body)
    )

    assert pa_regs == {x0, x1}
