from typing import cast

import pytest

from xdsl.dialects.arith import Addi, Arith, Constant, Subi
from xdsl.dialects.builtin import Builtin, IntegerAttr, IntegerType, ModuleOp, i32, i64
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.test import TestOp, TestTermOp
from xdsl.ir import Block, ErasedSSAValue, MLContext, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    Successor,
    VarRegion,
    irdl_op_definition,
    operand_def,
    successor_def,
    var_region_def,
)
from xdsl.parser import Parser
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


def test_ops_accessor():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region(block0)

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi(a, b)

    assert d.results[0] != c.results[0]

    assert c.lhs.owner is a
    assert c.rhs.owner is b
    assert d.lhs.owner is a
    assert d.rhs.owner is b


def test_ops_accessor_II():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region(block0)

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi(a, b)

    assert d.results[0] != c.results[0]

    # Erase operations and block
    region2 = Region()
    region.move_blocks(region2)

    region2.blocks[0].erase_op(a, safe_erase=False)
    region2.blocks[0].erase_op(b, safe_erase=False)
    region2.blocks[0].erase_op(c, safe_erase=False)

    assert isinstance(c.lhs, ErasedSSAValue)
    assert isinstance(c.rhs, ErasedSSAValue)
    assert c.lhs.owner is a
    assert c.rhs.owner is b

    region2.detach_block(block0)
    region2.drop_all_references()

    assert len(region2.blocks) == 0


def test_ops_accessor_III():
    # Create constants `from_attr` and add them, add them in blocks, blocks in
    # a region and create a function
    a = Constant.from_attr(IntegerAttr.from_int_and_width(1, 32), i32)
    b = Constant.from_attr(IntegerAttr.from_int_and_width(2, 32), i32)
    c = Constant.from_attr(IntegerAttr.from_int_and_width(3, 32), i32)
    d = Constant.from_attr(IntegerAttr.from_int_and_width(4, 32), i32)

    # Operation to add these constants
    e = Addi(a, b)
    f = Addi(c, d)

    # Create Blocks and Regions
    block0 = Block([a, b, e])
    block1 = Block([c, d, f])
    block2 = Block()

    region0 = Region([block0, block1])
    region1 = Region(block2)

    with pytest.raises(ValueError):
        region0.ops

    with pytest.raises(ValueError):
        region0.op

    with pytest.raises(ValueError):
        region1.op

    with pytest.raises(Exception):
        region1.detach_block(block0)

    region0.detach_block(block0)
    region0.detach_block(0)
    with pytest.raises(IndexError):
        region0.detach_block(1)


def test_op_operands_assign():
    """Test that we can directly assign `op.operands`."""
    val1, val2 = TestSSAValue(i32), TestSSAValue(i32)
    op = TestOp.create(operands=[val1, val2])
    op.operands = [val2, val1]
    op.verify()

    assert len(val1.uses) == 1
    assert len(val2.uses) == 1
    assert tuple(op.operands) == (val2, val1)


def test_op_operands_indexing():
    """Test `__getitem__`, `__setitem__`, and `__len__` on `op.operands`."""
    val1, val2 = TestSSAValue(i32), TestSSAValue(i32)
    op = TestOp.create(operands=[val1, val2])
    op.verify()

    assert op.operands[0] == val1
    assert op.operands[1] == val2
    assert op.operands[-1] == val2
    assert op.operands[0:2] == (val1, val2)

    op.operands[0] = val2
    op.verify()

    assert len(val1.uses) == 0
    assert len(val2.uses) == 2
    assert tuple(op.operands) == (val2, val2)


def test_op_clone():
    a = Constant.from_int_and_width(1, 32)
    b = a.clone()

    assert a is not b

    assert isinstance(b.value, IntegerAttr)
    b_value = cast(IntegerAttr[IntegerType], b.value)

    assert b_value.value.data == 1
    assert b_value.type.width.data == 32


def test_op_clone_with_regions():
    a = TestOp.create()
    op0 = TestOp.create(regions=[Region([Block([a])]), Region([Block([a.clone()])])])

    cloned_op = op0.clone()

    assert cloned_op is not op0
    assert len(cloned_op.regions[0].ops) == 1
    assert len(cloned_op.regions[1].ops) == 1

    for op0_region, cloned_op_region in zip(op0.regions, cloned_op.regions):
        for op0_region_op, cloned_region_op in zip(
            op0_region.ops, cloned_op_region.ops
        ):
            assert op0_region_op is not cloned_region_op


@irdl_op_definition
class SuccessorOp(IRDLOperation):
    """
    Utility operation that requires a successor.
    """

    name = "test.successor_op"

    successor: Successor = successor_def()

    traits = frozenset([IsTerminator()])


def test_block_branching_to_another_region_wrong():
    """
    Tests that an operation cannot have successors that branch to blocks of
    another region.
    """
    block1 = Block([TestOp.create(), TestOp.create()])
    region1 = Region([block1])

    op0 = TestOp.create(successors=[block1])
    block0 = Block([op0])
    region0 = Region([block0])
    region0 = TestOp.create(regions=[region0, region1])

    outer_block = Block([region0])

    with pytest.raises(
        VerifyException,
        match="is branching to a block of a different region",
    ):
        outer_block.verify()


def test_block_not_branching_to_another_region():
    """
    Tests that an operation can have successors that branch to blocks of the
    same region.
    """
    block0 = Block()

    op0 = SuccessorOp.create(successors=[block0])
    block1 = Block([op0])

    region0 = Region([block0, block1])

    region0.verify()


def test_empty_block_with_no_parent_region_requires_no_terminator():
    """
    Tests that an empty block belonging no parent region requires no terminator
    operation.
    """
    block0 = Block([])

    block0.verify()


def test_empty_block_with_orphan_single_block_parent_region_requires_no_terminator():
    """
    Tests that an empty block belonging to a single-block region with no parent
    operation requires no terminator operation.
    """
    block0 = Block([])
    region0 = Region([block0])

    region0.verify()


def test_empty_block_with_single_block_parent_region_requires_terminator():
    """
    Tests that an empty block belonging to a single-block region in a parent
    operation requires terminator operation.
    """
    block0 = Block([])
    region0 = Region([block0])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException,
        match="contains empty block in single-block region that expects at least a terminator",
    ):
        op0.verify()


def test_region_clone_into_circular_blocks():
    """
    Test that cloning a region with circular block dependency works.
    """
    region_str = """
    {
    ^0:
        "test.op"() [^1] : () -> ()
    ^1:
        "test.op"() [^0] : () -> ()
    }
    """
    ctx = MLContext(allow_unregistered=True)
    region = Parser(ctx, region_str).parse_region()

    region2 = Region()
    region.clone_into(region2)

    assert region.is_structurally_equivalent(region2)


def test_op_with_successors_not_in_block():
    block0 = Block()
    op0 = SuccessorOp.create(successors=[block0])

    with pytest.raises(
        VerifyException,
        match="with block successors does not belong to a block or a region",
    ):
        op0.verify()


def test_op_with_successors_not_in_region():
    block1 = Block()

    op0 = TestOp.create(successors=[block1])
    block0 = Block([op0])

    with pytest.raises(
        VerifyException,
        match="with block successors does not belong to a block or a region",
    ):
        block0.verify()


def test_non_empty_block_with_single_block_parent_region_must_have_terminator():
    """
    Tests that an non-empty block belonging to a single-block region with parent
    operation cannot have an operation that is not a terminator.
    """
    block1 = Block([TestOp.create()])
    region0 = Region([block1])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException,
        match="terminates block in single-block region but is not a terminator",
    ):
        op0.verify()


def test_non_empty_block_with_single_block_parent_region_with_terminator():
    """
    Tests that an non-empty block belonging to a single-block region with parent
    operation must have at least a terminator operation.
    """
    block0 = Block([TestTermOp.create()])
    region0 = Region([block0])
    op0 = TestOp.create(regions=[region0])

    op0.verify()


def test_non_empty_block_with_parent_region_can_have_terminator_with_successors():
    """
    Tests that an non-empty block belonging to a multi-block region with parent
    operation requires terminator operation.
    The terminator operation may have successors.
    """
    block0 = Block()
    block1 = Block([SuccessorOp.create(successors=[block0])])
    region0 = Region([block0, block1])
    op0 = TestOp.create(regions=[region0])

    op0.verify()


def test_non_empty_block_with_parent_region_requires_terminator_without_successors():
    """
    Tests that an non-empty block belonging to a multi-block region with parent
    operation requires terminator operation.
    The terminator operation may not have successors.
    """
    block0 = Block()
    block1 = Block([TestOp.create()])
    region0 = Region([block0, block1])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException,
        match="terminates block in multi-block region but is not a terminator",
    ):
        op0.verify()


def test_non_empty_block_with_parent_region_requires_terminator_with_successors():
    """
    Tests that an non-empty block belonging to a multi-block region with parent
    operation requires terminator operation.
    The terminator operation may have successors.
    """
    block0 = Block()

    op0 = TestOp.create(successors=[block0])
    block1 = Block([op0])

    region0 = Region([block0, block1])

    with pytest.raises(
        VerifyException,
        match="terminates block in multi-block region but is not a terminator",
    ):
        region0.verify()


def test_non_empty_block_with_parent_region_has_successors_but_not_last_block_op():
    """
    Tests that an non-empty block belonging to a multi-block region with parent
    operation requires terminator operation.
    """
    block0 = Block()
    block1 = Block([TestOp.create(successors=[block0]), TestOp.create()])
    region0 = Region([block0, block1])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException,
        match="with block successors must terminate its parent block",
    ):
        op0.verify()


##################### Testing is_structurally_equal #####################

program_region = """
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
}) : () -> ()
"""

program_region_2 = """
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 2 : i32} : () -> i32
}) : () -> ()
"""

program_region_2_diff_name = """
"builtin.module"() ({
  %cst = "arith.constant"() {"value" = 2 : i32} : () -> i32
}) : () -> ()
"""

program_region_2_diff_type = """
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 2 : i64} : () -> i64
}) : () -> ()
"""

program_add = """
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 2 : i32} : () -> i32
  %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
}) : () -> ()
"""

program_add_2 = """
"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 2 : i32} : () -> i32
  %2 = "arith.addi"(%1, %0) : (i32, i32) -> i32
}) : () -> ()
"""

program_func = """
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : i32, %1 : i32):
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
"""

program_successors = """
"builtin.module"() ({
  "func.func"() ({
  ^0:
    "cf.br"() [^1] : () -> ()
  ^1:
    "cf.br"() [^0] : () -> ()
  }) {"sym_name" = "unconditional_br", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
"""


@pytest.mark.parametrize(
    "args, expected_result",
    [
        ([program_region, program_region], True),
        ([program_region_2, program_region_2], True),
        ([program_region_2_diff_type, program_region_2_diff_type], True),
        ([program_region_2_diff_name, program_region_2_diff_name], True),
        ([program_region, program_region_2], False),
        ([program_region_2, program_region_2_diff_type], False),
        ([program_region_2, program_region_2_diff_name], True),
        ([program_add, program_add], True),
        ([program_add_2, program_add_2], True),
        ([program_add, program_add_2], False),
        ([program_func, program_func], True),
        ([program_successors, program_successors], True),
        ([program_successors, program_func], False),
        ([program_successors, program_add], False),
    ],
)
def test_is_structurally_equivalent(args: list[str], expected_result: bool):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)

    parser = Parser(ctx, args[0])
    lhs: Operation = parser.parse_op()

    parser = Parser(ctx, args[1])
    rhs: Operation = parser.parse_op()

    assert lhs.is_structurally_equivalent(rhs) == expected_result


def test_is_structurally_equivalent_free_operands():
    val1 = TestSSAValue(i32)
    val2 = TestSSAValue(i64)
    op1 = TestOp.create(operands=[val1, val2])
    op2 = TestOp.create(operands=[val1, val2])
    assert op1.is_structurally_equivalent(op2)


def test_is_structurally_equivalent_free_operands_fail():
    val1 = TestSSAValue(i32)
    val2 = TestSSAValue(i32)
    op1 = TestOp.create(operands=[val1])
    op2 = TestOp.create(operands=[val2])
    assert not op1.is_structurally_equivalent(op2)


def test_is_structurally_equivalent_incompatible_ir_nodes():
    program_func = """
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : i32, %1 : i32):
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    %3 = "arith.constant"() {"value" = 2 : i32} : () -> i32
    "func.return"(%3) : (i32) -> ()
  ^1(%4 : i32, %5 : i32):
    "func.return"(%4) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
"""
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)

    parser = Parser(ctx, program_func)
    program = parser.parse_operation()

    assert isinstance(program, ModuleOp)

    assert program.is_structurally_equivalent(program.regions[0]) is False
    assert program.is_structurally_equivalent(program.regions[0].blocks[0]) is False
    assert program.regions[0].is_structurally_equivalent(program) is False
    assert program.regions[0].blocks[0].is_structurally_equivalent(program) is False

    func_op = program.ops.first
    assert func_op is not None

    block = func_op.regions[0].blocks[0]
    ops = list(block.ops)
    assert not ops[0].is_structurally_equivalent(ops[1])
    assert not block.is_structurally_equivalent(func_op.regions[0].blocks[1])


def test_descriptions():
    a = Constant.from_int_and_width(1, 32)

    assert str(a.value) == "1 : i32"
    assert f"{a.value}" == "1 : i32"

    assert str(a) == "%0 = arith.constant 1 : i32"
    assert f"{a}" == "Constant(%0 = arith.constant 1 : i32)"

    m = ModuleOp([a])

    assert (
        str(m)
        == """\
builtin.module {
  %0 = arith.constant 1 : i32
}"""
    )

    assert (
        f"{m}"
        == """\
ModuleOp(
\tbuiltin.module {
\t  %0 = arith.constant 1 : i32
\t}
)"""
    )


# ToDo: Create this op without IRDL itself, since it tests fine grained
# stuff which is supposed to be used with IRDL or PDL.
@irdl_op_definition
class CustomOpWithMultipleRegions(IRDLOperation):
    name = "test.custom_op_with_multiple_regions"
    region: VarRegion = var_region_def()


def test_region_index_fetch():
    a = Constant.from_int_and_width(1, 32)
    b = Constant.from_int_and_width(2, 32)
    c = Constant.from_int_and_width(3, 32)
    d = Constant.from_int_and_width(4, 32)

    region0 = Region([Block([a])])
    region1 = Region([Block([b])])
    region2 = Region([Block([c])])
    region3 = Region([Block([d])])

    op = CustomOpWithMultipleRegions.build(
        regions=[[region0, region1, region2, region3]]
    )

    assert op.get_region_index(region0) == 0
    assert op.get_region_index(region1) == 1
    assert op.get_region_index(region2) == 2
    assert op.get_region_index(region3) == 3


def test_region_index_fetch_region_unavailability():
    a = Constant.from_int_and_width(1, 32)
    b = Constant.from_int_and_width(2, 32)

    region0 = Region([Block([a])])
    region1 = Region([Block([b])])

    op = CustomOpWithMultipleRegions.build(regions=[[region0]])

    assert op.get_region_index(region0) == 0
    with pytest.raises(Exception) as exc_info:
        op.get_region_index(region1)
    assert exc_info.value.args[0] == "Region is not attached to the operation."


def test_detach_region():
    a = Constant.from_int_and_width(1, 32)
    b = Constant.from_int_and_width(2, 32)
    c = Constant.from_int_and_width(3, 32)

    region0 = Region([Block([a])])
    region1 = Region([Block([b])])
    region2 = Region([Block([c])])

    op = CustomOpWithMultipleRegions.build(regions=[[region0, region1, region2]])

    assert op.detach_region(1) == region1
    assert op.detach_region(region0) == region0
    assert len(op.regions) == 1
    assert op.get_region_index(region2) == 0


@irdl_op_definition
class CustomVerify(IRDLOperation):
    name = "test.custom_verify_op"
    val: Operand = operand_def(i64)

    @staticmethod
    def get(val: SSAValue):
        return CustomVerify.build(operands=[val])

    def verify_(self):
        raise Exception("Custom Verification Check")


def test_op_custom_verify_is_called():
    a = Constant.from_int_and_width(1, i64)
    b = CustomVerify.get(a.result)
    with pytest.raises(Exception) as e:
        b.verify()
    assert e.value.args[0] == "Custom Verification Check"


def test_op_custom_verify_is_done_last():
    a = Constant.from_int_and_width(1, i32)
    # CustomVerify expects a i64, not i32
    b = CustomVerify.get(a.result)
    with pytest.raises(Exception) as e:
        b.verify()
    assert e.value.args[0] != "Custom Verification Check"
    assert "test.custom_verify_op operation does not verify" in e.value.args[0]


def test_block_walk():
    a = Constant.from_int_and_width(1, 32)
    b = Constant.from_int_and_width(2, 32)
    c = Constant.from_int_and_width(3, 32)

    ops = [a, b, c]
    block = Block(ops)

    assert list(block.walk()) == [a, b, c]
    assert list(block.walk_reverse()) == [c, b, a]


def test_region_walk():
    a = Constant.from_int_and_width(1, 32)
    b = Constant.from_int_and_width(2, 32)

    block_a = Block([a])
    block_b = Block([b])

    region = Region([block_a, block_b])

    assert list(region.walk()) == [a, b]
    assert list(region.walk_reverse()) == [b, a]


def test_op_walk():
    a = Constant.from_int_and_width(1, 32)
    b = Constant.from_int_and_width(2, 32)

    block_a = Block([a])
    block_b = Block([b])

    region_a = Region(block_a)
    region_b = Region(block_b)

    op_multi_region = TestOp.create(regions=[region_a, region_b])

    assert list(op_multi_region.walk()) == [op_multi_region, a, b]
    assert list(op_multi_region.walk_reverse()) == [b, a, op_multi_region]
