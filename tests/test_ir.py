import pytest

from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.arith import AddiOp, Arith, ConstantOp, SubiOp
from xdsl.dialects.builtin import (
    Builtin,
    Float64Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.ir import (
    Block,
    ErasedSSAValue,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    var_region_def,
)
from xdsl.parser import Parser
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


@irdl_op_definition
class TestWithPropOp(IRDLOperation):
    name = "test.op_with_prop"

    prop = prop_def()


def test_ops_accessor():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    # Operation to add these constants
    c = AddiOp(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region(block0)

    assert len(region.ops) == 3
    assert len(region.block.ops) == 3

    # Operation to subtract b from a
    d = SubiOp(a, b)

    assert d.results[0] != c.results[0]

    assert c.lhs.owner is a
    assert c.rhs.owner is b
    assert d.lhs.owner is a
    assert d.rhs.owner is b


def test_ops_accessor_II():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    # Operation to add these constants
    c = AddiOp(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region(block0)

    assert len(region.ops) == 3
    assert len(region.block.ops) == 3

    # Operation to subtract b from a
    d = SubiOp(a, b)

    assert d.results[0] != c.results[0]

    # Erase operations and block
    region2 = Region()
    region.move_blocks(region2)

    region2.block.erase_op(a, safe_erase=False)
    region2.block.erase_op(b, safe_erase=False)
    region2.block.erase_op(c, safe_erase=False)

    assert isinstance(c.lhs, ErasedSSAValue)
    assert isinstance(c.rhs, ErasedSSAValue)
    assert c.lhs.owner is a
    assert c.rhs.owner is b

    region2.detach_block(block0)
    region2.drop_all_references()

    assert len(region2.blocks) == 0


def test_ops_accessor_III():
    # Create constants and add them, add them in blocks, blocks in
    # a region and create a function
    a = ConstantOp(IntegerAttr.from_int_and_width(1, 32), i32)
    b = ConstantOp(IntegerAttr.from_int_and_width(2, 32), i32)
    c = ConstantOp(IntegerAttr.from_int_and_width(3, 32), i32)
    d = ConstantOp(IntegerAttr.from_int_and_width(4, 32), i32)

    # Operation to add these constants
    e = AddiOp(a, b)
    f = AddiOp(c, d)

    # Create Blocks and Regions
    block0 = Block([a, b, e])
    block1 = Block([c, d, f])
    block2 = Block()

    region0 = Region([block0, block1])
    region1 = Region(block2)

    with pytest.raises(
        ValueError,
        match="'ops' property of Region class is only available for single-block regions.",
    ):
        region0.ops

    with pytest.raises(
        ValueError,
        match="'op' property of Region class is only available for single-operation single-block regions.",
    ):
        region0.op

    with pytest.raises(
        ValueError,
        match="'op' property of Region class is only available for single-operation single-block regions.",
    ):
        region1.op

    with pytest.raises(
        ValueError,
        match="Block is not a child of the region.",
    ):
        region1.detach_block(block0)

    region0.detach_block(block0)
    region0.detach_block(0)
    with pytest.raises(IndexError):
        region0.detach_block(1)


def test_op_operands_assign():
    """Test that we can directly assign `op.operands`."""
    val1, val2 = create_ssa_value(i32), create_ssa_value(i32)
    op = test.TestOp.create(operands=[val1, val2])
    op.operands = [val2, val1]
    op.verify()

    assert val1.has_one_use()
    assert val2.has_one_use()
    assert tuple(op.operands) == (val2, val1)


def test_op_operands_indexing():
    """Test `__getitem__`, `__setitem__`, and `__len__` on `op.operands`."""
    val1, val2 = create_ssa_value(i32), create_ssa_value(i32)
    op = test.TestOp.create(operands=[val1, val2])
    op.verify()

    assert op.operands[0] == val1
    assert op.operands[1] == val2
    assert op.operands[-1] == val2
    assert op.operands[0:2] == (val1, val2)

    op.operands[0] = val2
    op.verify()

    assert not list(val1.uses)
    assert len(list(val2.uses)) == 2
    assert tuple(op.operands) == (val2, val2)


def test_op_operands_comparison():
    """Test `__eq__`, and `__hash__` on `op.operands`."""
    val1, val2 = create_ssa_value(i32), create_ssa_value(i32)
    op1 = test.TestOp.create(operands=[val1, val2])
    op2 = test.TestOp.create(operands=[val1, val2])
    op1.verify()
    op2.verify()

    assert op1.operands == op2.operands
    assert hash(op1.operands) == hash(op2.operands)

    op1.operands[0] = val2
    op1.verify()

    assert op1.operands != op2.operands


def test_op_clone():
    a = TestWithPropOp.create(
        properties={"prop": i32}, attributes={"attr": i64}, result_types=(i32,)
    )
    a.results[0].name_hint = "name_hint"
    b = a.clone()

    assert a is not b
    assert a.is_structurally_equivalent(b)

    # Name hints

    c = a.clone(clone_name_hints=True)
    d = a.clone(clone_name_hints=False)
    assert a is not c
    assert a is not d

    assert a.is_structurally_equivalent(c)
    assert a.is_structurally_equivalent(d)

    assert b.results[0].name_hint == "name_hint"
    assert c.results[0].name_hint == "name_hint"
    assert d.results[0].name_hint is None


def test_op_clone_with_regions():
    # Children
    ca0 = test.TestOp.create(result_types=(i32,))
    ca0.results[0].name_hint = "a"
    ca1 = test.TestOp.create(result_types=(i32,))
    ca1.results[0].name_hint = "b"
    # Parent
    pa = test.TestOp.create(
        regions=[
            Region([Block([ca0], arg_types=(i32,))]),
            Region([Block([ca1], arg_types=(i32,))]),
        ]
    )
    pa.regions[0].block.args[0].name_hint = "ca0"
    pa.regions[1].block.args[0].name_hint = "ca1"

    pb = pa.clone()
    assert pa is not pb

    assert len(pb.regions[0].ops) == 1
    assert len(pb.regions[1].ops) == 1

    for op0_region, cloned_op_region in zip(pa.regions, pb.regions):
        for op0_region_op, cloned_region_op in zip(
            op0_region.ops, cloned_op_region.ops
        ):
            assert op0_region_op is not cloned_region_op

    pc = pa.clone(clone_name_hints=True)
    pd = pa.clone(clone_name_hints=False)

    def name_hints(op: Operation):
        for o in op.walk():
            for res in o.results:
                yield res.name_hint
            for r in o.regions:
                for b in r.blocks:
                    for arg in b.args:
                        yield arg.name_hint

    assert tuple(name_hints(pa)) == ("ca0", "ca1", "a", "b")
    assert tuple(name_hints(pb)) == ("ca0", "ca1", "a", "b")
    assert tuple(name_hints(pc)) == ("ca0", "ca1", "a", "b")
    assert tuple(name_hints(pd)) == (None, None, None, None)


def test_op_clone_graph_region():
    # Children
    ca0 = test.TestOp.create(result_types=(i32,))
    ca0.results[0].name_hint = "a"
    ca1 = test.TestOp.create(result_types=(i32,))
    ca1.results[0].name_hint = "b"
    # Make recursive
    ca0.operands = (ca1.results[0],)
    ca1.operands = (ca0.results[0],)
    # Parent
    pa = test.TestOp.create(regions=[Region([Block([ca0, ca1])])])

    pb = pa.clone()
    assert pa is not pb

    cb0, cb1 = pb.regions[0].blocks[0].ops

    assert ca0 is not cb0
    assert ca1 is not cb1
    assert ca0.operands != cb0.operands
    assert ca1.operands != cb1.operands


def test_op_clone_wrong_block_order():
    # Children
    ca1 = test.TestOp.create(result_types=(i32,))
    ca1.results[0].name_hint = "b"
    ca0 = test.TestOp.create(operands=ca1.results, result_types=(i32,))
    ca0.results[0].name_hint = "a"
    # Parent
    pa = test.TestOp.create(regions=[Region([Block([ca0]), Block([ca1])])])

    pb = pa.clone()
    assert pa is not pb

    (cb0,) = pb.regions[0].blocks[0].ops
    (cb1,) = pb.regions[0].blocks[1].ops

    assert ca0 is not cb0
    assert ca1 is not cb1
    for oa, ob in zip(ca0.operands, cb0.operands, strict=True):
        assert oa is not ob
    for oa, ob in zip(ca1.operands, cb1.operands, strict=True):
        assert oa is not ob


def test_block_branching_to_another_region_wrong():
    """
    Tests that an operation cannot have successors that branch to blocks of
    another region.
    """

    block1 = Block([test.TestOp.create(), test.TestOp.create()])
    region1 = Region([block1])

    op0 = test.TestOp.create(successors=[block1])
    block0 = Block([op0])
    region0 = Region([block0])
    region0 = test.TestOp.create(regions=[region0, region1])

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

    op0 = test.TestTermOp.create(successors=[block0])
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
    op0 = test.TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException,
        match="contains empty block in single-block region that expects at least a terminator",
    ):
        op0.verify()


def test_split_block_first():
    old_block = Block((test.TestOp(), test.TestOp(), test.TestOp()))
    region = Region(old_block)
    a, b, c = old_block.ops

    # Check preconditions

    assert old_block.parent is region
    assert list(region.blocks) == [old_block]

    assert old_block.first_op is a
    assert old_block.last_op is c

    assert a.parent is old_block
    assert b.parent is old_block
    assert c.parent is old_block

    assert a.next_op is b
    assert b.next_op is c
    assert c.next_op is None

    assert a.prev_op is None
    assert b.prev_op is a
    assert c.prev_op is b

    new_block = old_block.split_before(a)

    # Check postconditions

    assert old_block.parent is region
    assert new_block.parent is region
    assert list(region.blocks) == [old_block, new_block]

    assert old_block.first_op is None
    assert old_block.last_op is None
    assert new_block.first_op is a
    assert new_block.last_op is c

    assert a.parent is new_block
    assert b.parent is new_block
    assert c.parent is new_block

    assert a.next_op is b
    assert b.next_op is c
    assert c.next_op is None

    assert a.prev_op is None
    assert b.prev_op is a
    assert c.prev_op is b


def test_split_block_middle():
    old_block = Block((test.TestOp(), test.TestOp(), test.TestOp()))
    region = Region(old_block)
    a, b, c = old_block.ops

    # Check preconditions

    assert old_block.parent is region
    assert list(region.blocks) == [old_block]

    assert old_block.first_op is a
    assert old_block.last_op is c

    assert a.parent is old_block
    assert b.parent is old_block
    assert c.parent is old_block

    assert a.next_op is b
    assert b.next_op is c
    assert c.next_op is None

    assert a.prev_op is None
    assert b.prev_op is a
    assert c.prev_op is b

    new_block = old_block.split_before(b)

    # Check postconditions

    assert old_block.parent is region
    assert new_block.parent is region
    assert list(region.blocks) == [old_block, new_block]

    assert old_block.first_op is a
    assert old_block.last_op is a
    assert new_block.first_op is b
    assert new_block.last_op is c

    assert a.parent is old_block
    assert b.parent is new_block
    assert c.parent is new_block

    assert a.next_op is None
    assert b.next_op is c
    assert c.next_op is None

    assert a.prev_op is None
    assert b.prev_op is None
    assert c.prev_op is b


def test_split_block_last():
    old_block = Block((test.TestOp(), test.TestOp(), test.TestOp()))
    region = Region(old_block)
    a, b, c = old_block.ops

    # Check preconditions

    assert old_block.parent is region
    assert list(region.blocks) == [old_block]

    assert old_block.first_op is a
    assert old_block.last_op is c

    assert a.parent is old_block
    assert b.parent is old_block
    assert c.parent is old_block

    assert a.next_op is b
    assert b.next_op is c
    assert c.next_op is None

    assert a.prev_op is None
    assert b.prev_op is a
    assert c.prev_op is b

    new_block = old_block.split_before(c)

    # Check postconditions

    assert old_block.parent is region
    assert new_block.parent is region
    assert list(region.blocks) == [old_block, new_block]

    assert old_block.first_op is a
    assert old_block.last_op is b
    assert new_block.first_op is c
    assert new_block.last_op is c

    assert a.parent is old_block
    assert b.parent is old_block
    assert c.parent is new_block

    assert a.next_op is b
    assert b.next_op is None
    assert c.next_op is None

    assert a.prev_op is None
    assert b.prev_op is a
    assert c.prev_op is None


def test_split_block_args():
    old_block = Block((test.TestOp(), test.TestOp(), test.TestOp()))
    region = Region(old_block)
    _, op, _ = region.block.ops

    new_block = old_block.split_before(op, arg_types=(i32, i64))

    arg_types = new_block.arg_types
    assert arg_types == (i32, i64)


def test_region_clone_into_circular_blocks():
    """
    Test that cloning a region with circular block dependency works.
    """
    region_str = """
    {
    ^bb0:
        "test.op"() [^bb1] : () -> ()
    ^bb1:
        "test.op"() [^bb0] : () -> ()
    }
    """
    ctx = Context(allow_unregistered=True)
    region = Parser(ctx, region_str).parse_region()

    region2 = Region()
    region.clone_into(region2)

    assert region.is_structurally_equivalent(region2)


def test_op_with_successors_not_in_block():
    block0 = Block()
    op0 = test.TestTermOp.create(successors=[block0])

    with pytest.raises(
        VerifyException,
        match="with block successors does not belong to a block or a region",
    ):
        op0.verify()


def test_op_with_successors_not_in_region():
    block1 = Block()

    op0 = test.TestOp.create(successors=[block1])
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
    block1 = Block([test.TestOp.create()])
    region0 = Region([block1])
    op0 = test.TestOp.create(regions=[region0])

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
    block0 = Block([test.TestTermOp.create()])
    region0 = Region([block0])
    op0 = test.TestOp.create(regions=[region0])

    op0.verify()


def test_non_empty_block_with_parent_region_can_have_terminator_with_successors():
    """
    Tests that an non-empty block belonging to a multi-block region with parent
    operation requires terminator operation.
    The terminator operation may have successors.
    """
    block0 = Block()
    block1 = Block([test.TestTermOp.create(successors=[block0])])
    region0 = Region([block0, block1])
    op0 = test.TestOp.create(regions=[region0])

    op0.verify()


def test_non_empty_block_with_parent_region_requires_terminator_without_successors():
    """
    Tests that an non-empty block belonging to a multi-block region with parent
    operation requires terminator operation.
    The terminator operation may not have successors.
    """
    block0 = Block()
    block1 = Block([test.TestOp.create()])
    region0 = Region([block0, block1])
    op0 = test.TestOp.create(regions=[region0])

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

    op0 = test.TestOp.create(successors=[block0])
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
    block1 = Block([test.TestOp.create(successors=[block0]), test.TestOp.create()])
    region0 = Region([block0, block1])
    op0 = test.TestOp.create(regions=[region0])

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
  ^bb0(%0 : i32, %1 : i32):
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
"""

program_successors = """
"builtin.module"() ({
  "func.func"() ({
  ^bb0:
    "cf.br"() [^bb1] : () -> ()
  ^bb1:
    "cf.br"() [^bb0] : () -> ()
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
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Cf)

    parser = Parser(ctx, args[0])
    lhs: Operation = parser.parse_op()

    parser = Parser(ctx, args[1])
    rhs: Operation = parser.parse_op()

    assert lhs.is_structurally_equivalent(rhs) == expected_result


def test_is_structurally_equivalent_properties():
    op32 = TestWithPropOp.create(properties={"prop": i32})
    op32prime = TestWithPropOp.create(properties={"prop": i32})
    op64 = TestWithPropOp.create(properties={"prop": i64})
    assert op32.is_structurally_equivalent(op32prime)
    assert not op32.is_structurally_equivalent(op64)


def test_is_structurally_equivalent_free_operands():
    val1 = create_ssa_value(i32)
    val2 = create_ssa_value(i64)
    op1 = test.TestOp.create(operands=[val1, val2])
    op2 = test.TestOp.create(operands=[val1, val2])
    assert op1.is_structurally_equivalent(op2)


def test_is_structurally_equivalent_free_operands_fail():
    val1 = create_ssa_value(i32)
    val2 = create_ssa_value(i32)
    op1 = test.TestOp.create(operands=[val1])
    op2 = test.TestOp.create(operands=[val2])
    assert not op1.is_structurally_equivalent(op2)


def test_is_structurally_equivalent_incompatible_ir_nodes():
    program_func = """
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%0 : i32, %1 : i32):
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    %3 = "arith.constant"() {"value" = 2 : i32} : () -> i32
    "func.return"(%3) : (i32) -> ()
  ^bb1(%4 : i32, %5 : i32):
    "func.return"(%4) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
"""
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Cf)

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
    a = ConstantOp.from_int_and_width(1, 32)

    assert str(a.value) == "1 : i32"
    assert f"{a.value}" == "1 : i32"

    assert str(a) == "%0 = arith.constant 1 : i32"
    assert f"{a}" == "ConstantOp(%0 = arith.constant 1 : i32)"

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
class MultipleRegionsOp(IRDLOperation):
    name = "test.custom_op_with_multiple_regions"
    region = var_region_def()


def test_region_index_fetch():
    a = ConstantOp.from_int_and_width(1, 32)
    b = ConstantOp.from_int_and_width(2, 32)
    c = ConstantOp.from_int_and_width(3, 32)
    d = ConstantOp.from_int_and_width(4, 32)

    region0 = Region([Block([a])])
    region1 = Region([Block([b])])
    region2 = Region([Block([c])])
    region3 = Region([Block([d])])

    op = MultipleRegionsOp.build(regions=[[region0, region1, region2, region3]])

    assert op.get_region_index(region0) == 0
    assert op.get_region_index(region1) == 1
    assert op.get_region_index(region2) == 2
    assert op.get_region_index(region3) == 3


def test_region_index_fetch_region_unavailability():
    a = ConstantOp.from_int_and_width(1, 32)
    b = ConstantOp.from_int_and_width(2, 32)

    region0 = Region([Block([a])])
    region1 = Region([Block([b])])

    op = MultipleRegionsOp.build(regions=[[region0]])

    assert op.get_region_index(region0) == 0
    with pytest.raises(ValueError, match="Region is not attached to the operation."):
        op.get_region_index(region1)


def test_detach_region():
    a = ConstantOp.from_int_and_width(1, 32)
    b = ConstantOp.from_int_and_width(2, 32)
    c = ConstantOp.from_int_and_width(3, 32)

    region0 = Region([Block([a])])
    region1 = Region([Block([b])])
    region2 = Region([Block([c])])

    op = MultipleRegionsOp.build(regions=[[region0, region1, region2]])

    assert op.detach_region(1) == region1
    assert op.detach_region(region0) == region0
    assert len(op.regions) == 1
    assert op.get_region_index(region2) == 0


def test_region_hashable():
    a = Region()
    b = Region()
    assert a == a
    assert a != b
    assert hash(a) == hash(a)
    assert a in {a}
    assert b not in {a}


@irdl_op_definition
class CustomVerifyOp(IRDLOperation):
    name = "test.custom_verify_op"
    val = operand_def(i64)

    @staticmethod
    def get(val: SSAValue):
        return CustomVerifyOp.build(operands=[val])

    def verify_(self):
        raise Exception("Custom Verification Check")


def test_op_custom_verify_is_called():
    a = ConstantOp.from_int_and_width(1, i64)
    b = CustomVerifyOp.get(a.result)
    with pytest.raises(Exception, match="Custom Verification Check"):
        b.verify()


def test_op_custom_verify_is_done_last():
    a = ConstantOp.from_int_and_width(1, i32)
    # CustomVerify expects a i64, not i32
    b = CustomVerifyOp.get(a.result)
    with pytest.raises(
        VerifyException,
        match="operand 'val' at position 0 does not verify:\nExpected attribute i64 but got i32",
    ):
        b.verify()


def test_block_walk():
    a = ConstantOp.from_int_and_width(1, 32)
    b = ConstantOp.from_int_and_width(2, 32)
    c = ConstantOp.from_int_and_width(3, 32)

    ops = [a, b, c]
    block = Block(ops)

    assert list(block.walk()) == [a, b, c]
    assert list(block.walk(reverse=True)) == [c, b, a]


def test_region_walk():
    a = ConstantOp.from_int_and_width(1, 32)
    b = ConstantOp.from_int_and_width(2, 32)

    block_a = Block([a])
    block_b = Block([b])

    region = Region([block_a, block_b])

    assert list(region.walk()) == [a, b]
    assert list(region.walk(reverse=True)) == [b, a]


def test_op_walk():
    a = ConstantOp.from_int_and_width(1, 32)
    b = ConstantOp.from_int_and_width(2, 32)

    block_a = Block([a])
    block_b = Block([b])

    region_a = Region(block_a)
    region_b = Region(block_b)

    op_multi_region = test.TestOp.create(regions=[region_a, region_b])

    assert list(op_multi_region.walk()) == [op_multi_region, a, b]
    assert list(op_multi_region.walk(reverse=True)) == [op_multi_region, b, a]
    assert list(op_multi_region.walk(region_first=True)) == [a, b, op_multi_region]
    assert list(op_multi_region.walk(region_first=True, reverse=True)) == [
        b,
        a,
        op_multi_region,
    ]


def test_region_clone():
    a = ConstantOp.from_int_and_width(1, 32)
    block_a = Block([a])
    region = Region(block_a)
    region2 = region.clone()
    assert region.is_structurally_equivalent(region2)


def test_get_attr_or_prop():
    a = test.TestOp.create(
        attributes={"attr": StringAttr("attr"), "attr_and_prop": StringAttr("attr")},
        properties={"prop": StringAttr("prop"), "attr_and_prop": StringAttr("prop")},
    )
    assert a.get_attr_or_prop("attr") == StringAttr("attr")
    assert a.get_attr_or_prop("prop") == StringAttr("prop")
    assert a.get_attr_or_prop("attr_and_prop") == StringAttr("prop")
    assert a.get_attr_or_prop("none") is None


def test_dialect_name():
    class MyOperation(Operation):
        name = "dialect.op"

    assert MyOperation.dialect_name() == "dialect"


def test_replace_by_if():
    a = create_ssa_value(i32)
    b = test.TestOp((a,))
    c = test.TestOp((a,))

    assert set(u.operation for u in a.uses) == {b, c}

    d = create_ssa_value(i32)
    a.replace_by_if(d, lambda u: u.operation is not c)

    assert set(u.operation for u in a.uses) == {c}
    assert set(u.operation for u in d.uses) == {b}


def test_same_block():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1, op2]))])

    assert op1.is_before_in_block(op2)
    assert not op2.is_before_in_block(op1)
    assert not op1.is_before_in_block(op1)


def test_different_blocks():
    op1 = test.TestOp()
    op2 = test.TestOp()
    test.TestOp(regions=[Region(Block([op1])), Region(Block([op2]))])

    assert not op1.is_before_in_block(op2)
    assert not op2.is_before_in_block(op1)


def test_ancestor_block_in_region_nested():
    # lvl 3
    op_3 = test.TestOp()

    # lvl 2
    blk_2 = Block([op_3])
    op_2 = test.TestOp(regions=[Region(blk_2)])

    # lvl 1
    blk_1 = Block([op_2])
    reg_1 = Region(blk_1)
    test.TestOp(regions=[reg_1])

    assert reg_1.find_ancestor_block_in_region(blk_2) is blk_1
    assert reg_1.find_ancestor_block_in_region(blk_1) is blk_1


def test_ancestor_block_in_region_different_region():
    # lvl 3
    op_3 = test.TestOp()

    # lvl 2
    blk_2 = Block([op_3])
    op_2 = test.TestOp(regions=[Region(blk_2)])

    # lvl 1
    blk_1 = Block([op_2])
    reg_1_1 = Region(blk_1)
    reg_1_2 = Region()
    test.TestOp(regions=[reg_1_1, reg_1_2])

    assert reg_1_2.find_ancestor_block_in_region(blk_2) is None


def test_find_ancestor_op_in_block():
    op1 = test.TestOp()
    op2 = test.TestOp(regions=[Region(Block([op1]))])
    op3 = test.TestOp(regions=[Region(Block([op2]))])

    blk_top = Block([op3])
    op4 = test.TestOp(regions=[Region(blk_top)])

    assert blk_top.find_ancestor_op_in_block(op1) is op3
    assert blk_top.find_ancestor_op_in_block(op4) is None


def test_ssa_get_on_ssa():
    ssa_value = create_ssa_value(i32)

    assert SSAValue.get(ssa_value) == ssa_value
    assert SSAValue.get(ssa_value, type=IntegerType) == ssa_value
    assert SSAValue.get(ssa_value, type=IntegerType).type == i32

    with pytest.raises(
        ValueError,
        match="SSAValue.get: Expected <class 'xdsl.dialects.builtin.IndexType'> but got SSAValue with type i32",
    ):
        SSAValue.get(ssa_value, type=IndexType)


def test_ssa_get_with_typeform():
    ssa_value = create_ssa_value(i32)

    assert SSAValue.get(ssa_value, type=IntegerType | IndexType) == ssa_value
    assert SSAValue.get(ssa_value, type=IntegerType | IndexType).type == i32

    with pytest.raises(
        ValueError,
        match="SSAValue.get: Expected xdsl.dialects.builtin.Float64Type | xdsl.dialects.builtin.IndexType but got SSAValue with type i32",
    ):
        SSAValue.get(ssa_value, type=Float64Type | IndexType)

    tensor_type = TensorType(i32, [2, 2])
    ssa_value_tensor = create_ssa_value(tensor_type)

    assert (
        SSAValue.get(ssa_value_tensor, type=TensorType[IntegerType]) == ssa_value_tensor
    )
    assert (
        SSAValue.get(ssa_value_tensor, type=TensorType[IntegerType]).type == tensor_type
    )

    assert (
        SSAValue.get(ssa_value_tensor, type=TensorType[IntegerType | IndexType])
        == ssa_value_tensor
    )
    assert (
        SSAValue.get(ssa_value_tensor, type=TensorType[IntegerType | IndexType]).type
        == tensor_type
    )

    with pytest.raises(
        ValueError,
        match=r"SSAValue.get: Expected xdsl.dialects.builtin.TensorType\[xdsl.dialects.builtin.Float64Type \| xdsl.dialects.builtin.IndexType\] but got SSAValue with type tensor<2x2xi32>.",
    ):
        SSAValue.get(ssa_value_tensor, type=TensorType[Float64Type | IndexType])


def test_ssa_get_on_op():
    op1 = test.TestOp(result_types=[i32])

    assert SSAValue.get(op1).owner == op1
    assert SSAValue.get(op1).type == i32
    assert SSAValue.get(op1, type=IntegerType).owner == op1
    assert SSAValue.get(op1, type=IntegerType).type == i32

    with pytest.raises(
        ValueError,
        match="SSAValue.get: Expected <class 'xdsl.dialects.builtin.IndexType'> but got SSAValue with type i32",
    ):
        SSAValue.get(op1, type=IndexType)

    op2 = test.TestOp(result_types=[i32, i32])
    with pytest.raises(
        ValueError, match="SSAValue.get: expected operation with a single result."
    ):
        SSAValue.get(op2)


def test_repr():
    bb = tuple(Block(arg_types=(i32,)) for _ in range(3))
    bb[0].add_op(test.TestTermOp(result_types=(i32,), successors=bb[1:]))
    bb[1].add_op(
        test.TestTermOp(operands=bb[0].args, result_types=(i32,), successors=bb[2:])
    )
    bb[2].add_op(test.TestTermOp(operands=bb[0].args, result_types=(i32,)))
    root = test.TestOp(regions=(r := Region(bb),))

    ops = tuple(root.walk())
    assert len(ops) == 4
    regions = tuple(region for op in ops for region in op.regions)
    assert len(regions) == 1
    blocks = tuple(block for region in regions for block in region.blocks)
    assert len(blocks) == 3
    results = tuple(result for op in ops for result in op.results)
    assert len(results) == 3
    args = tuple(arg for block in blocks for arg in block.args)
    assert len(args) == 3
    values = (*results, *args)
    block_uses = tuple(use for block in blocks for use in block.uses)
    assert len(block_uses) == 3
    value_uses = tuple(use for value in values for use in value.uses)
    assert len(value_uses) == 2

    blocks[0].args[0].name_hint = "bb0arg"
    ops[1].results[0].name_hint = "op1res"

    op_reprs = tuple(repr(op) for op in ops)
    assert op_reprs == (
        f"<TestOp {id(root)}(operands=[], results=[], successors=[], properties={{}}, attributes={{}}, regions=[<Region {id(r)}>], parent=None, _next_op=None, _prev_op=None)>",
        f"<TestTermOp {id(ops[1])}(operands=[], results=[<OpResult {id(results[0])}>], successors=[<Block {id(bb[1])}>, <Block {id(bb[2])}>], properties={{}}, attributes={{}}, regions=[], parent=<Block {id(bb[0])}>, _next_op=None, _prev_op=None)>",
        f"<TestTermOp {id(ops[2])}(operands=[<BlockArgument {id(args[0])}>], results=[<OpResult {id(results[1])}>], successors=[<Block {id(bb[2])}>], properties={{}}, attributes={{}}, regions=[], parent=<Block {id(bb[1])}>, _next_op=None, _prev_op=None)>",
        f"<TestTermOp {id(ops[3])}(operands=[<BlockArgument {id(args[0])}>], results=[<OpResult {id(results[2])}>], successors=[], properties={{}}, attributes={{}}, regions=[], parent=<Block {id(bb[2])}>, _next_op=None, _prev_op=None)>",
    )

    region_reprs = tuple(repr(region) for region in regions)
    assert region_reprs == (
        f"Region(blocks=[<Block {id(blocks[0])}>, <Block {id(blocks[1])}>, <Block {id(blocks[2])}>])",
    )

    block_reprs = tuple(repr(block) for block in blocks)
    assert block_reprs == (
        f"<Block {id(blocks[0])}(_args=(<BlockArgument[i32] name_hint: bb0arg, index: 0, uses: 2>,), num_ops=1)>",
        f"<Block {id(blocks[1])}(_args=(<BlockArgument[i32] name_hint: None, index: 0, uses: 0>,), num_ops=1)>",
        f"<Block {id(blocks[2])}(_args=(<BlockArgument[i32] name_hint: None, index: 0, uses: 0>,), num_ops=1)>",
    )

    op_result_reprs = tuple(repr(result) for result in results)
    assert op_result_reprs == (
        "<OpResult[i32] name_hint: op1res, index: 0, operation: test.termop, uses: 0>",
        "<OpResult[i32] name_hint: None, index: 0, operation: test.termop, uses: 0>",
        "<OpResult[i32] name_hint: None, index: 0, operation: test.termop, uses: 0>",
    )

    block_arg_reprs = tuple(repr(arg) for arg in args)
    assert block_arg_reprs == (
        "<BlockArgument[i32] name_hint: bb0arg, index: 0, uses: 2>",
        "<BlockArgument[i32] name_hint: None, index: 0, uses: 0>",
        "<BlockArgument[i32] name_hint: None, index: 0, uses: 0>",
    )

    block_use_reprs = tuple(repr(use) for use in block_uses)
    assert block_use_reprs == (
        f"<Use {id(block_uses[0])}(_operation=<TestTermOp {id(ops[1])}>, _index=0, _prev_use=None, _next_use=None)>",
        f"<Use {id(block_uses[1])}(_operation=<TestTermOp {id(ops[2])}>, _index=0, _prev_use=None, _next_use=<Use {id(block_uses[2])}>)>",
        f"<Use {id(block_uses[2])}(_operation=<TestTermOp {id(ops[1])}>, _index=1, _prev_use=<Use {id(block_uses[1])}>, _next_use=None)>",
    )

    value_use_reprs = tuple(repr(use) for use in value_uses)
    assert value_use_reprs == (
        f"<Use {id(value_uses[0])}(_operation=<TestTermOp {id(ops[3])}>, _index=0, _prev_use=None, _next_use=<Use {id(value_uses[1])}>)>",
        f"<Use {id(value_uses[1])}(_operation=<TestTermOp {id(ops[2])}>, _index=0, _prev_use=<Use {id(value_uses[0])}>, _next_use=None)>",
    )
