import pytest

from typing import cast, Annotated

from xdsl.dialects.arith import Arith, Addi, Subi, Constant
from xdsl.dialects.builtin import Builtin, IntegerType, i32, i64, IntegerAttr, ModuleOp
from xdsl.dialects.func import Func
from xdsl.dialects.cf import Cf
from xdsl.dialects.scf import If

from xdsl.ir import MLContext, Operation, Block, Region, ErasedSSAValue, SSAValue
from xdsl.parser import XDSLParser
from xdsl.irdl import IRDLOperation, VarRegion, irdl_op_definition, Operand


def test_ops_accessor():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region(block0)

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi.get(a, b)

    assert d.results[0] != c.results[0]

    assert c.lhs.owner is a
    assert c.rhs.owner is b
    assert d.lhs.owner is a
    assert d.rhs.owner is b


def test_ops_accessor_II():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block([a, b, c])
    # Create a region to include a, b, c
    region = Region(block0)

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi.get(a, b)

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
    e = Addi.get(a, b)
    f = Addi.get(c, d)

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


def test_op_clone():
    a = Constant.from_int_and_width(1, 32)
    b = a.clone()

    assert a is not b

    assert isinstance(b.value, IntegerAttr)
    b_value = cast(IntegerAttr[IntegerType], b.value)

    assert b_value.value.data == 1
    assert b_value.typ.width.data == 32


def test_op_clone_with_regions():
    cond = Constant.from_int_and_width(1, 1)
    a = Constant.from_int_and_width(1, 32)
    if_ = If.get(cond, [], Region([Block([a])]), Region([Block([a.clone()])]))

    if2 = if_.clone()

    assert if2 is not if_
    assert len(if2.true_region.ops) == 1
    assert len(if2.false_region.ops) == 1
    assert if2.true_region.op is not if_.true_region.op
    assert if2.false_region.op is not if_.false_region.op


##################### Testing is_structurally_equal #####################

program_region = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
}
"""
program_region_2 = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 2 : !i32]
}
"""
program_region_2_diff_name = """builtin.module() {
  %cst : !i32 = arith.constant() ["value" = 2 : !i32]
}
"""
program_region_2_diff_type = """builtin.module() {
  %0 : !i64 = arith.constant() ["value" = 2 : !i64]
}
"""
program_add = """builtin.module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
}
"""
program_add_2 = """builtin.module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
}
"""
program_func = """builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
program_successors = """
    func.func() ["sym_name" = "unconditional_br", "function_type" = !fun<[], []>, "sym_visibility" = "private"] {
    ^0:
        cf.br() (^1)
    ^1:
        cf.br() (^0)
    }
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

    parser = XDSLParser(ctx, args[0])
    lhs: Operation = parser.parse_op()

    parser = XDSLParser(ctx, args[1])
    rhs: Operation = parser.parse_op()

    assert lhs.is_structurally_equivalent(rhs) == expected_result


def test_is_structurally_equivalent_incompatible_ir_nodes():
    program_func = """builtin.module() {
    func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
    ^0(%0 : !i32, %1 : !i32):
      %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
      %3 : !i32 = arith.constant() ["value" = 2 : !i32]
      func.return(%3 : !i32)
    ^1(%4 : !i32, %5 : !i32):
      func.return(%4 : !i32)
    }
  }
  """
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)

    parser = XDSLParser(ctx, program_func)
    program = parser.parse_operation()

    assert isinstance(program, ModuleOp)

    assert program.is_structurally_equivalent(program.regions[0]) == False
    assert program.is_structurally_equivalent(program.regions[0].blocks[0]) == False
    assert program.regions[0].is_structurally_equivalent(program) == False
    assert program.regions[0].blocks[0].is_structurally_equivalent(program) == False
    assert (
        program.ops[0]
        .regions[0]
        .blocks[0]
        .ops[0]
        .is_structurally_equivalent(program.ops[0].regions[0].blocks[0].ops[1])
        == False
    )
    assert (
        program.ops[0]
        .regions[0]
        .blocks[0]
        .is_structurally_equivalent(program.ops[0].regions[0].blocks[1])
        == False
    )


def test_descriptions():
    a = Constant.from_int_and_width(1, 32)

    assert str(a.value) == "1 : i32"
    assert f"{a.value}" == "1 : i32"

    assert str(a) == '%0 : !i32 = arith.constant() ["value" = 1 : !i32]'
    assert f"{a}" == 'Constant(%0 : !i32 = arith.constant() ["value" = 1 : !i32])'

    m = ModuleOp([a])

    assert (
        str(m)
        == """\
builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
}"""
    )

    assert (
        f"{m}"
        == """\
ModuleOp(
\tbuiltin.module() {
\t  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
\t}
)"""
    )


# ToDo: Create this op without IRDL itself, since it tests fine grained
# stuff which is supposed to be used with IRDL or PDL.
@irdl_op_definition
class CustomOpWithMultipleRegions(IRDLOperation):
    name = "test.custom_op_with_multiple_regions"
    region: VarRegion


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
    val: Annotated[Operand, i64]

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
    assert (
        e.value.args[0]
        == "test.custom_verify_op operation does not verify\n\ntest.custom_verify_op(%<UNKNOWN> : !i32)\n\n"
    )


def test_replace_operand():
    cst0 = Constant.from_int_and_width(0, 32).result
    cst1 = Constant.from_int_and_width(1, 32).result
    add = Addi.get(cst0, cst1)

    new_cst = Constant.from_int_and_width(2, 32).result
    add.replace_operand(cst0, new_cst)

    assert new_cst in add.operands
    assert cst0 not in add.operands

    with pytest.raises(ValueError):
        add.replace_operand(cst0, new_cst)
