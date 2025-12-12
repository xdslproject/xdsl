from collections.abc import Sequence

from xdsl import ir, irdl
from xdsl.backend.register_type import (
    RegisterAllocatedMemoryEffect,
    RegisterResource,
    RegisterType,
)
from xdsl.traits import (
    EffectInstance,
    MemoryEffectKind,
    get_effects,
    is_side_effect_free,
)


@irdl.irdl_attr_definition
class TestRegister(RegisterType):
    name = "test.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return {"x0": 0, "x1": 1, "a0": 0, "a1": 1}

    @classmethod
    def infinite_register_prefix(cls):
        return "y"


@irdl.irdl_op_definition
class TestAllocatableOp(irdl.IRDLOperation):
    name = "test.allocatable"

    in_operands = irdl.var_operand_def()
    inout_operands = irdl.var_operand_def()
    out_results = irdl.var_result_def()
    inout_results = irdl.var_result_def()

    traits = irdl.traits_def(RegisterAllocatedMemoryEffect())

    irdl_options = [
        irdl.AttrSizedOperandSegments(),
        irdl.AttrSizedResultSegments(),
    ]

    def __init__(
        self,
        in_operands: Sequence[ir.SSAValue],
        inout_operands: Sequence[ir.SSAValue],
        out_result_types: Sequence[ir.Attribute],
        inout_result_types: Sequence[ir.Attribute],
    ):
        super().__init__(
            operands=(in_operands, inout_operands),
            result_types=(out_result_types, inout_result_types),
        )


def test_register_resource_name():
    """Test that RegisterResource returns a descriptive name."""
    reg = TestRegister.from_name("x0")
    resource = RegisterResource(reg)
    assert "<Register !test.reg<x0>>" == resource.name()


def test_no_effects_for_unallocated_registers():
    """Test that unallocated registers produce no memory effects."""
    op = TestAllocatableOp([], [], [TestRegister.unallocated()], [])
    effects = get_effects(op)
    assert effects == set()
    assert is_side_effect_free(op)


def test_write_effect_for_allocated_result():
    """Test that allocated register results produce WRITE effects."""
    allocated_reg = TestRegister.from_name("x0")
    op = TestAllocatableOp([], [], [allocated_reg], [])
    effects = get_effects(op)

    assert effects == {
        EffectInstance(MemoryEffectKind.WRITE, resource=RegisterResource(allocated_reg))
    }


def test_read_effect_for_allocated_operand():
    """Test that allocated register operands produce READ effects."""
    # Create an op that produces an allocated register to use as operand
    allocated_reg = TestRegister.from_name("x1")
    producer = TestAllocatableOp([], [], [allocated_reg], [])

    # Use the result as an operand
    op = TestAllocatableOp([producer.results[0]], [], [], [])
    effects = get_effects(op)

    assert effects == {
        EffectInstance(MemoryEffectKind.READ, resource=RegisterResource(allocated_reg))
    }


def test_mixed_read_write_effects():
    """Test an op with both allocated operands and results."""
    reg_x0 = TestRegister.from_name("x0")
    reg_x1 = TestRegister.from_name("x1")

    # Producer for operand
    producer = TestAllocatableOp([], [], [reg_x0], [])

    # Op that reads x0 and writes x1
    op = TestAllocatableOp([producer.results[0]], [], [reg_x1], [])
    effects = get_effects(op)

    assert effects == {
        EffectInstance(MemoryEffectKind.READ, resource=RegisterResource(reg_x0)),
        EffectInstance(MemoryEffectKind.WRITE, resource=RegisterResource(reg_x1)),
    }
