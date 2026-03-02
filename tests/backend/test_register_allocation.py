import re
from collections.abc import Sequence
from typing import ClassVar

import pytest

from xdsl.backend.block_naive_allocator import BlockNaiveAllocator
from xdsl.backend.register_allocatable import (
    HasRegisterConstraints,
    RegisterAllocatableOperation,
    RegisterConstraints,
)
from xdsl.backend.register_allocator import ValueAllocator
from xdsl.backend.register_stack import OutOfRegisters, RegisterStack
from xdsl.backend.register_type import RegisterType
from xdsl.builder import Builder
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute, Block, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    IRDLOperation,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.test_value import create_ssa_value


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

    irdl_options = (AttrSizedOperandSegments(), AttrSizedResultSegments())

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


def test_new_type_for_value():
    @irdl_attr_definition
    class OtherRegister(RegisterType):
        name = "test.other_reg"

        @classmethod
        def index_by_name(cls) -> dict[str, int]:
            return {"x0": 0, "x1": 1, "a0": 0, "a1": 1}

        @classmethod
        def infinite_register_prefix(cls):
            return "y"

    u = TestRegister.unallocated()
    a0 = TestRegister.from_name("a0")
    a1 = TestRegister.from_name("a1")

    available_registers = RegisterStack.get((a0, a1))
    allocator = ValueAllocator(available_registers, TestRegister)

    r0, r1, r2, r3 = op(
        (), u, a0, OtherRegister.unallocated(), OtherRegister.from_index(1)
    ).results

    assert allocator.new_type_for_value(r0) == a1
    assert allocator.new_type_for_value(r1) is None
    assert allocator.new_type_for_value(r2) is None
    assert allocator.new_type_for_value(r3) is None


def test_allocate_value():
    u = TestRegister.unallocated()
    a0 = TestRegister.from_name("a0")
    a1 = TestRegister.from_name("a1")
    y0 = TestRegister.from_name("y0")

    register_stack = RegisterStack.get((a0, a1), allow_infinite=True)
    allocator = ValueAllocator(register_stack, TestRegister)

    op0 = op((), u, u, u, u)
    r00, r01, r02, _r03 = op0.results
    op1 = op((r00, r01), u, u, inouts=(r02,))
    results00 = tuple(op0.results)
    results10 = tuple(op1.results)

    # Initial state
    assert register_stack.available_registers == {"test.reg": [0, 1]}
    assert allocator.new_value_by_old_value == {}
    assert op0.result_types == (u, u, u, u)
    assert op1.result_types == (u, u, u)

    # Allocate first result
    new_value = allocator.allocate_value(op1.out_results[0])
    assert new_value is not None

    results11 = tuple(op1.results)

    assert register_stack.available_registers == {"test.reg": [0]}
    assert allocator.new_value_by_old_value == {results10[0]: results11[0]}
    assert op0.result_types == (u, u, u, u)
    assert op1.result_types == (a1, u, u)

    # Allocate first result again (old value) -> no change of state
    new_value = allocator.allocate_value(results10[0])
    assert new_value is None
    assert register_stack.available_registers == {"test.reg": [0]}
    assert allocator.new_value_by_old_value == {results10[0]: results11[0]}
    assert op0.result_types == (u, u, u, u)
    assert op1.result_types == (a1, u, u)

    # Allocate first result again (new value) -> no change of state
    new_value = allocator.allocate_value(op1.out_results[0])
    assert new_value is None
    assert register_stack.available_registers == {"test.reg": [0]}
    assert allocator.new_value_by_old_value == {results10[0]: results11[0]}
    assert op0.result_types == (u, u, u, u)
    assert op1.result_types == (a1, u, u)

    # Allocate second result
    new_value = allocator.allocate_value(op1.out_results[1])
    assert new_value is not None

    results12 = tuple(op1.results)

    assert register_stack.available_registers == {"test.reg": []}
    assert allocator.new_value_by_old_value == {
        results10[0]: results11[0],
        results11[1]: results12[1],
    }
    assert op0.result_types == (u, u, u, u)
    assert op1.result_types == (a1, a0, u)

    # Allocate inout result and operands at the same time
    new_value = allocator.allocate_values_same_reg(
        (op1.inout_operands[0], op1.inout_results[0])
    )
    results03 = tuple(op0.results)
    results13 = tuple(op1.results)

    assert register_stack.available_registers == {"test.reg": []}
    assert allocator.new_value_by_old_value == {
        results10[0]: results11[0],
        results11[1]: results12[1],
        results12[2]: results13[2],
        results00[2]: results03[2],
    }
    assert op0.result_types == (u, u, y0, u)
    assert op1.result_types == (a1, a0, y0)

    # Free the allocated result values
    allocator.free_value(results13[0])
    allocator.free_value(results13[1])

    assert register_stack.available_registers == {"test.reg": [1, 0]}
    assert allocator.new_value_by_old_value == {
        results10[0]: results11[0],
        results11[1]: results12[1],
        results12[2]: results13[2],
        results00[2]: results03[2],
    }
    assert op0.result_types == (u, u, y0, u)
    assert op1.result_types == (a1, a0, y0)

    # Allocate operand
    allocator.allocate_value(op1.operands[0])
    results04 = tuple(op0.results)

    assert register_stack.available_registers == {"test.reg": [1]}
    assert allocator.new_value_by_old_value == {
        results10[0]: results11[0],
        results11[1]: results12[1],
        results12[2]: results13[2],
        results00[2]: results03[2],
        results00[0]: results04[0],
    }
    assert op0.result_types == (a0, u, y0, u)
    assert op1.result_types == (a1, a0, y0)


def test_allocate_values_same_reg():
    u = TestRegister.unallocated()
    a0 = TestRegister.from_name("a0")
    a1 = TestRegister.from_name("a1")
    y0 = TestRegister.from_name("y0")

    register_stack = RegisterStack.get((a0, a1))
    allocator = ValueAllocator(register_stack, TestRegister)

    # Empty
    assert not allocator.allocate_values_same_reg(())

    op0 = op((), u, u, u, u, u)

    # 1 unallocated
    assert allocator.allocate_values_same_reg((op0.results[0],))
    assert op0.result_types == (a1, u, u, u, u)

    # 1 allocated
    assert not allocator.allocate_values_same_reg((op0.results[0],))
    assert op0.result_types == (a1, u, u, u, u)

    # 2 unallocated
    assert allocator.allocate_values_same_reg((op0.results[1], op0.results[2]))
    assert op0.result_types == (a1, a0, a0, u, u)

    # 1 allocated 1 unallocated
    assert allocator.allocate_values_same_reg((op0.results[0], op0.results[3]))
    assert op0.result_types == (a1, a0, a0, a1, u)

    # 1 unallocated 1 allocated
    assert allocator.allocate_values_same_reg((op0.results[4], op0.results[0]))
    assert op0.result_types == (a1, a0, a0, a1, a1)

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!test.reg<a0>', '!test.reg<a1>']"
        ),
    ):
        allocator.allocate_values_same_reg((op0.results[0], op0.results[1]))

    op1 = op((), y0)

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!test.reg<a0>', '!test.reg<a1>', '!test.reg<y0>']"
        ),
    ):
        allocator.allocate_values_same_reg(
            (op0.results[0], op0.results[1], op1.results[0])
        )


def test_multiple_outputs():
    class AllocatableTestOp(TestOp, HasRegisterConstraints):
        def get_register_constraints(self) -> RegisterConstraints:
            # The default register constraints are that all operands are "in", and all
            # results are "out" registers.
            return RegisterConstraints(self.operands, self.results, ())

    available_registers = RegisterStack(allow_infinite=True)
    register_allocator = BlockNaiveAllocator(available_registers, TestRegister)

    op = AllocatableTestOp(
        result_types=(
            TestRegister.unallocated(),
            TestRegister.unallocated(),
        )
    )

    op.allocate_registers(register_allocator)

    # Check allocated registers are unique
    assert len(op.result_types) == len(set(op.result_types))


def test_fail_error_message():
    @irdl_op_definition
    class InoutOp(HasRegisterConstraints, IRDLOperation):
        name = "test.inout"

        T: ClassVar = VarConstraint("T", base(TestRegister))

        a = operand_def(T)
        b = result_def(T)

        def get_register_constraints(self) -> RegisterConstraints:
            return RegisterConstraints((), (), ((self.a, self.b),))

    a = create_ssa_value(TestRegister.from_index(0))
    block = Block((InoutOp(operands=[a], result_types=[TestRegister.from_index(1)]),))

    allocator = BlockNaiveAllocator(RegisterStack.get(), TestRegister)

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!test.reg<a0>', '!test.reg<a1>']"
        ),
    ) as e:
        allocator.allocate_block(block)

    assert getattr(e.value, "__notes__") == [
        """
^bb0:
  %0 = "test.inout"(%1) : (!test.reg<a0>) -> !test.reg<a1>
  ^^^^^^^^^^^^^^^^^----
  | Error allocating op
  ---------------------
"""
    ]


def test_out_of_registers():
    register_stack = RegisterStack()
    with pytest.raises(OutOfRegisters, match="Out of registers."):
        register_stack.pop(TestRegister)
