from __future__ import annotations

import re

import pytest

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import DenseArrayBase, StringAttr, i32
from xdsl.dialects.test import TestTermOp
from xdsl.ir import Block, Region
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    AttrSizedSuccessorSegments,
    IRDLOperation,
    SameVariadicOperandSize,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    opt_region_def,
    opt_result_def,
    opt_successor_def,
    prop_def,
    region_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
    var_successor_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

################################################################################
#                                 Results                                      #
################################################################################


@irdl_op_definition
class ResultOp(IRDLOperation):
    name = "test.result_op"

    res = result_def(StringAttr)


def test_result_builder():
    op = ResultOp.build(result_types=[StringAttr("")])
    op.verify()
    assert op.result_types == (StringAttr(""),)


def test_result_builder_exception():
    with pytest.raises(ValueError, match="Expected 1 result, but got 0"):
        ResultOp.build()


@irdl_op_definition
class OptResultOp(IRDLOperation):
    name = "test.opt_result_op"

    res = opt_result_def(StringAttr)


def test_opt_result_builder():
    op1 = OptResultOp.build(result_types=[[StringAttr("")]])
    op2 = OptResultOp.build(result_types=[[]])
    op3 = OptResultOp.build(result_types=[None])
    op1.verify()
    op2.verify()
    op3.verify()
    assert op1.result_types == (StringAttr(""),)
    assert len(op2.results) == 0
    assert len(op3.results) == 0


def test_opt_result_builder_two_args():
    with pytest.raises(
        ValueError,
        match=(
            "Error in test.opt_result_op builder: optional VarIRConstruct.RESULT 0 "
            "'res' expects a list of size at most 1 or None, but got a list of size 2"
        ),
    ):
        OptResultOp.build(result_types=[[StringAttr(""), StringAttr("")]])


@irdl_op_definition
class VarResultOp(IRDLOperation):
    name = "test.var_result_op"

    res = var_result_def(StringAttr)


def test_var_result_builder():
    op = VarResultOp.build(result_types=[[StringAttr("0"), StringAttr("1")]])
    op.verify()
    assert op.result_types == (
        StringAttr("0"),
        StringAttr("1"),
    )


@irdl_op_definition
class TwoVarResultOp(IRDLOperation):
    name = "test.two_var_result_op"

    res1 = var_result_def(StringAttr)
    res2 = var_result_def(StringAttr)
    irdl_options = (AttrSizedResultSegments(),)


def test_two_var_result_builder():
    op = TwoVarResultOp.build(
        result_types=[
            [StringAttr("0"), StringAttr("1")],
            [StringAttr("2"), StringAttr("3")],
        ]
    )
    op.verify()
    assert op.result_types == (
        StringAttr("0"),
        StringAttr("1"),
        StringAttr("2"),
        StringAttr("3"),
    )

    assert op.attributes[
        AttrSizedResultSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [2, 2])


def test_two_var_result_builder2():
    op = TwoVarResultOp.build(
        result_types=[
            [StringAttr("0")],
            [StringAttr("1"), StringAttr("2"), StringAttr("3")],
        ]
    )
    op.verify()
    assert op.result_types == (
        StringAttr("0"),
        StringAttr("1"),
        StringAttr("2"),
        StringAttr("3"),
    )
    assert op.attributes[
        AttrSizedResultSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [1, 3])


@irdl_op_definition
class MixedResultOp(IRDLOperation):
    name = "test.mixed"

    res1 = var_result_def(StringAttr)
    res2 = result_def(StringAttr)
    res3 = var_result_def(StringAttr)
    irdl_options = (AttrSizedResultSegments(),)


def test_var_mixed_builder():
    op = MixedResultOp.build(
        result_types=[
            [StringAttr("0"), StringAttr("1")],
            StringAttr("2"),
            [StringAttr("3"), StringAttr("4")],
        ]
    )
    op.verify()
    assert op.result_types == (
        StringAttr("0"),
        StringAttr("1"),
        StringAttr("2"),
        StringAttr("3"),
        StringAttr("4"),
    )

    assert op.attributes[
        AttrSizedResultSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [2, 1, 2])


################################################################################
#                                 Operands                                     #
################################################################################


@irdl_op_definition
class OperandOp(IRDLOperation):
    name = "test.operand_op"

    res = operand_def(StringAttr)


def test_operand_builder_operation():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = OperandOp.build(operands=[op1])
    op2.verify()
    assert tuple(op2.operands) == (op1.res,)


def test_operand_builder_value():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = OperandOp.build(operands=[op1.res])
    op2.verify()
    assert tuple(op2.operands) == (op1.res,)


def test_operand_builder_exception():
    with pytest.raises(ValueError, match="Expected 1 operand, but got 0"):
        OperandOp.build()


@irdl_op_definition
class OptOperandOp(IRDLOperation):
    name = "test.opt_operand_op"

    res = opt_operand_def(StringAttr)


def test_opt_operand_builder():
    op = ResultOp.build(result_types=[StringAttr("0")])
    op1 = OptOperandOp.build(operands=[[op]])
    op2 = OptOperandOp.build(operands=[[]])
    op1.verify()
    op2.verify()
    assert [operand for operand in op1.operands] == [op.res]
    assert len(op2.operands) == 0


def test_opt_operand_builder_two_args():
    op = ResultOp.build(result_types=[StringAttr("0")])
    with pytest.raises(
        ValueError,
        match=(
            "Error in test.opt_operand_op builder: optional VarIRConstruct.OPERAND 0 "
            "'res' expects a list of size at most 1 or None, but got a list of size 2"
        ),
    ):
        OptOperandOp.build(operands=[[op, op]])


@irdl_op_definition
class VarOperandOp(IRDLOperation):
    name = "test.var_operand_op"

    res = var_operand_def(StringAttr)


def test_var_operand_builder():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = VarOperandOp.build(operands=[[op1, op1]])
    op2.verify()
    assert tuple(op2.operands) == (op1.res, op1.res)


@irdl_op_definition
class TwoVarOperandOp(IRDLOperation):
    name = "test.two_var_operand_op"

    res1 = var_operand_def(StringAttr)
    res2 = var_operand_def(StringAttr)
    irdl_options = (AttrSizedOperandSegments(),)


# Define a similar operation with the segment sizes as a property to test this case
@irdl_op_definition
class TwoVarOperandPropOp(IRDLOperation):
    name = "test.two_var_operand_op"

    res1 = var_operand_def(StringAttr)
    res2 = var_operand_def(StringAttr)
    irdl_options = (AttrSizedOperandSegments(as_property=True),)


def test_two_var_operand_builder():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = TwoVarOperandOp.build(operands=[[op1, op1], [op1, op1]])
    op2.verify()
    assert tuple(op2.operands) == (op1.res, op1.res, op1.res, op1.res)
    assert op2.attributes[
        AttrSizedOperandSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [2, 2])


def test_two_var_operand_builder2():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = TwoVarOperandOp.build(operands=[[op1], [op1, op1, op1]])
    op2.verify()
    assert tuple(op2.operands) == (op1.res, op1.res, op1.res, op1.res)
    assert op2.attributes[
        AttrSizedOperandSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [1, 3])


def test_two_var_operand_prop_builder():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = TwoVarOperandPropOp.build(operands=[[op1, op1], [op1, op1]])
    op2.verify()
    assert tuple(op2.operands) == (op1.res, op1.res, op1.res, op1.res)
    assert op2.properties[
        AttrSizedOperandSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [2, 2])


def test_two_var_operand_prop_builder2():
    op1 = ResultOp.build(result_types=[StringAttr("0")])
    op2 = TwoVarOperandPropOp.build(operands=[[op1], [op1, op1, op1]])
    op2.verify()
    assert tuple(op2.operands) == (op1.res, op1.res, op1.res, op1.res)
    assert op2.properties[
        AttrSizedOperandSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [1, 3])


@irdl_op_definition
class SameSizeVarOperandOp(IRDLOperation):
    name = "test.same_size_var_operand_op"

    var1 = var_operand_def()
    op1 = operand_def()
    var2 = var_operand_def()
    irdl_options = (SameVariadicOperandSize(),)


def test_same_size_operand_builder():
    op1 = ResultOp.build(result_types=[StringAttr("0")]).res
    op2 = SameSizeVarOperandOp.build(operands=[[op1, op1], op1, [op1, op1]])
    op2.verify()
    assert tuple(op2.operands) == (op1, op1, op1, op1, op1)
    op2 = SameSizeVarOperandOp.create(operands=[op1, op1, op1, op1, op1])
    op2.verify()
    assert (op2.var1, op2.op1, op2.var2) == ((op1, op1), op1, (op1, op1))


def test_same_size_operand_builder2():
    op1 = ResultOp.build(result_types=[StringAttr("0")]).res
    with pytest.raises(
        ValueError, match=re.escape("Variadic operands have different sizes: [1, 3]")
    ):
        SameSizeVarOperandOp.build(operands=[[op1], op1, [op1, op1, op1]])
    op2 = SameSizeVarOperandOp.create(operands=[op1, op1, op1, op1])
    with pytest.raises(
        VerifyException,
        match=re.escape(
            "Operation does not verify: Operation has 3 operands for 2 variadic operands marked as having the same size."
        ),
    ):
        op2.verify()


################################################################################
#                                Attribute                                     #
################################################################################


@irdl_op_definition
class AttrOp(IRDLOperation):
    name = "test.attr_op"
    attr = attr_def(StringAttr)


def test_attr_op():
    op = AttrOp.build(attributes={"attr": StringAttr("0")})
    op.verify()
    assert op.attr == StringAttr("0")


def test_attr_new_attr_op():
    op = AttrOp.build(attributes={"attr": StringAttr("0"), "new_attr": StringAttr("1")})
    op.verify()
    assert op.attr == StringAttr("0")
    assert op.attributes["new_attr"] == StringAttr("1")


@irdl_op_definition
class OptionalAttrOp(IRDLOperation):
    name = "test.opt_attr_op"

    opt_attr = opt_attr_def(StringAttr)


def test_optional_attr_op_empty():
    op = OptionalAttrOp.build()
    op.verify()
    assert op.opt_attr is None


################################################################################
#                                 Property                                     #
################################################################################


@irdl_op_definition
class PropertyOp(IRDLOperation):
    name = "test.prop_op"
    attr = prop_def(StringAttr)


def test_prop_op():
    op = PropertyOp.build(properties={"attr": StringAttr("0")})
    op.verify()
    assert op.attr == StringAttr("0")


@irdl_op_definition
class OptionalPropertyOp(IRDLOperation):
    name = "test.opt_prop_op"

    opt_attr = opt_prop_def(StringAttr)


def test_optional_prop_op_empty():
    op = OptionalPropertyOp.build()
    op.verify()
    assert op.opt_attr is None


################################################################################
#                                  Region                                      #
################################################################################


@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "test.region_op"

    region = region_def()


def test_region_op_region():
    op = RegionOp.build(regions=[Region()])
    op.verify()

    assert not op.region.blocks


def test_region_op_blocks():
    op = RegionOp.build(regions=[[Block(), Block()]])
    op.verify()
    assert len(op.region.blocks) == 2


def test_region_op_ops():
    op1 = RegionOp.build(regions=[[Block([TestTermOp.create()])]])
    op2 = RegionOp.build(regions=[[Block([TestTermOp.create()])]])
    op = RegionOp.build(regions=[[op1, op2, TestTermOp.create()]])
    op.verify()
    assert len(op.region.blocks) == 1
    assert len(op.region.blocks[0].ops) == 3


def test_noop_region():
    region0 = Region(Block())
    assert len(region0.ops) == 0


def test_singleop_region():
    a = ConstantOp.from_int_and_width(1, i32)
    region0 = Region(Block([a]))
    assert type(region0.op) is ConstantOp


@irdl_op_definition
class SBRegionOp(IRDLOperation):
    name = "test.sbregion_op"

    region = region_def("single_block")


def test_sbregion_one_block():
    op = SBRegionOp.build(regions=[[Block([TestTermOp.create()])]])
    op.verify()
    assert len(op.region.blocks) == 1


@irdl_op_definition
class OptRegionOp(IRDLOperation):
    name = "test.opt_region_op"

    reg = opt_region_def()


def test_opt_region_builder():
    op1 = OptRegionOp.build(regions=[[[Block(), Block()]]])
    op2 = OptRegionOp.build(regions=[[Region()]])
    op3 = OptRegionOp.build(regions=[None])
    op1.verify()
    op2.verify()
    op3.verify()


def test_opt_region_builder_two_args():
    with pytest.raises(
        ValueError,
        match=(
            "Error in test.opt_region_op builder: optional VarIRConstruct.REGION 0 "
            "'reg' expects a list of size at most 1 or None, but got a list of size 2"
        ),
    ):
        OptRegionOp.build(regions=[[Region(), Region()]])


@irdl_op_definition
class OptSBRegionOp(IRDLOperation):
    name = "test.sbregion_op"

    region = opt_region_def("single_block")


def test_opt_sbregion_one_block():
    op1 = OptSBRegionOp.build(regions=[[[Block([TestTermOp.create()])]]])
    op2 = OptSBRegionOp.build(regions=[[]])
    op1.verify()
    op2.verify()
    assert op1.region is not None
    assert len(op1.region.blocks) == 1
    assert op2.region is None


@irdl_op_definition
class VarRegionOp(IRDLOperation):
    name = "test.var_operand_op"

    regs = var_region_def()


def test_var_region_builder():
    op = VarRegionOp.build(regions=[[Region(), [Block(), Block()]]])
    op.verify()
    assert len(op.regs[0].blocks) == 0
    assert len(op.regs[1].blocks) == 2


@irdl_op_definition
class VarSBRegionOp(IRDLOperation):
    name = "test.sbregion_op"

    regs = var_region_def("single_block")


def test_var_sbregion_one_block():
    op1 = VarSBRegionOp.build(regions=[[[Block([TestTermOp.create()])]]])
    op2 = VarSBRegionOp.build(regions=[[Region(), [Block(), Block()]]])
    op1.verify()
    op2.verify()
    assert len(op1.regs) == 1
    assert len(op2.regs) == 2
    assert len(op2.regs[0].blocks) == 0
    assert len(op2.regs[1].blocks) == 2


@irdl_op_definition
class TwoVarRegionOp(IRDLOperation):
    name = "test.two_var_region_op"

    res1 = var_region_def()
    res2 = var_region_def()
    irdl_options = (AttrSizedRegionSegments(),)


def test_two_var_region_builder():
    region1 = Region()
    region2 = Region()
    region3 = Region()
    region4 = Region()
    op2 = TwoVarRegionOp.build(regions=[[region1, region2], [region3, region4]])
    op2.verify()
    assert op2.regions == (region1, region2, region3, region4)
    assert op2.attributes[
        AttrSizedRegionSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [2, 2])


def test_two_var_operand_builder3():
    region1 = Region()
    region2 = Region()
    region3 = Region()
    region4 = Region()
    op2 = TwoVarRegionOp.build(regions=[[region1], [region2, region3, region4]])
    op2.verify()
    assert op2.regions == (region1, region2, region3, region4)
    assert op2.attributes[
        AttrSizedRegionSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [1, 3])


################################################################################
#                                Successor                                     #
################################################################################


@irdl_op_definition
class SuccessorOp(IRDLOperation):
    name = "test.successor_op"

    successor = successor_def()

    traits = traits_def(IsTerminator())


def test_successor_op_successor():
    """Test operation from IRDL operation definition can have successors"""
    block0 = Block()

    op = SuccessorOp.build(successors=[block0])
    block1 = Block([op])

    _ = Region([block0, block1])

    op.verify()
    assert len(op.successors) == 1


@irdl_op_definition
class OptSuccessorOp(IRDLOperation):
    name = "test.opt_successora_op"

    successor = opt_successor_def()

    traits = traits_def(IsTerminator())


def test_opt_successor_builder():
    """
    Test operation from IRDL operation definition can optionally have
    successors
    """
    block0 = Block()

    op1 = OptSuccessorOp.build(successors=[block0])
    block1 = Block([op1])

    op2 = OptSuccessorOp.build(successors=[None])
    block2 = Block([op2])

    _ = Region([block0, block1, block2])

    op1.verify()
    op2.verify()


@irdl_op_definition
class VarSuccessorOp(IRDLOperation):
    name = "test.var_succesor_op"

    successor = var_successor_def()

    traits = traits_def(IsTerminator())


def test_var_successor_builder():
    """
    Test operation from IRDL operation definition can have variadic successors
    """
    block0 = Block()
    op = VarSuccessorOp.build(successors=[[block0, block0, block0]])

    block1 = Block([op])
    _ = Region([block0, block1])

    op.verify()
    assert len(op.successors) == 3


@irdl_op_definition
class TwoVarSuccessorOp(IRDLOperation):
    name = "test.two_var_successor_op"

    res1 = var_successor_def()
    res2 = var_successor_def()
    irdl_options = (AttrSizedSuccessorSegments(),)

    traits = traits_def(IsTerminator())


def test_two_var_successor_builder():
    """
    Test operation from IRDL operation definition can have variadic successors
    along with their sizes as an attribute
    """
    block1 = Block()
    block2 = Block()
    block3 = Block()
    block4 = Block()

    op2 = TwoVarSuccessorOp.build(successors=[[block1, block2], [block3, block4]])
    block0 = Block([op2])

    _ = Region([block0, block1, block2, block3, block4])

    op2.verify()
    assert tuple(op2.successors) == (block1, block2, block3, block4)
    assert op2.attributes[
        AttrSizedSuccessorSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [2, 2])


def test_two_var_successor_builder2():
    """
    Test operation from IRDL operation definition can have variadic successors
    along with their sizes as an attribute
    """
    block1 = Block()
    block2 = Block()
    block3 = Block()
    block4 = Block()

    op2 = TwoVarSuccessorOp.build(successors=[[block1], [block2, block3, block4]])
    block0 = Block([op2])

    _ = Region([block0, block1, block2, block3, block4])

    op2.verify()
    assert tuple(op2.successors) == (block1, block2, block3, block4)
    assert op2.attributes[
        AttrSizedSuccessorSegments.attribute_name
    ] == DenseArrayBase.from_list(i32, [1, 3])


################################################################################
#                                   Misc                                       #
################################################################################


def test_parent_pointers():
    op = ResultOp.build(result_types=[StringAttr("0")])
    block = Block([op])
    reg = Region(block)
    reg_op = RegionOp.build(regions=[reg])

    block_2 = Block([reg_op])
    reg_2 = Region(block_2)
    reg_op_2 = RegionOp.build(regions=[reg_2])

    assert op.parent_block() is block
    assert op.parent_region() is reg
    assert op.parent_op() is reg_op

    assert reg_op_2.parent_block() is None
    assert reg_op_2.parent_region() is None
    assert reg_op_2.parent_op() is None

    assert reg.parent_op() is reg_op
    assert reg.parent_block() is block_2
    assert reg.parent_region() is reg_2

    assert reg_2.parent_block() is None
    assert reg_2.parent_region() is None

    assert block.parent_region() is reg
    assert block.parent_op() is reg_op
    assert block.parent_block() is block_2

    assert block_2.parent_block() is None
