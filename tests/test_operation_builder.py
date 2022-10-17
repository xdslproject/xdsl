from __future__ import annotations
from typing import Annotated, Optional
import pytest

from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, VectorType,
                                   IntegerType, Operation, StringAttr)
from xdsl.ir import Block, OpResult, SSAValue
from xdsl.irdl import (OptOperandDef, OptRegionDef, OptResultDef,
                       OptSingleBlockRegionDef, SingleBlockRegionDef,
                       VarRegionDef, VarSingleBlockRegionDef,
                       irdl_op_definition, ResultDef, VarResultDef,
                       AttrSizedResultSegments, OperandDef, VarOperandDef,
                       AttrSizedOperandSegments, AttributeDef, RegionDef,
                       OptAttributeDef, Region)

#  ____                 _ _
# |  _ \ ___  ___ _   _| | |_
# | |_) / _ \/ __| | | | | __|
# |  _ <  __/\__ \ |_| | | |_
# |_| \_\___||___/\__,_|_|\__|
#


@irdl_op_definition
class ResultOp(Operation):
    name: str = "test.result_op"

    res: Annotated[OpResult, ResultDef(StringAttr)]


def test_result_builder():
    op = ResultOp.build(result_types=[0])
    op.verify()
    assert [res.typ for res in op.results] == [StringAttr.from_int(0)]


def test_result_builder_exception():
    with pytest.raises(ValueError):
        ResultOp.build()


@irdl_op_definition
class OptResultOp(Operation):
    name: str = "test.opt_result_op"

    res: Annotated[Optional[OpResult], OptResultDef(StringAttr)]


def test_opt_result_builder():
    op1 = OptResultOp.build(result_types=[[0]])
    op2 = OptResultOp.build(result_types=[[]])
    op1.verify()
    op2.verify()
    assert [res.typ for res in op1.results] == [StringAttr.from_int(0)]
    assert len(op2.results) == 0


def test_opt_result_builder_two_args():
    with pytest.raises(ValueError) as _:
        OptResultOp.build(result_types=[[0, 0]])


@irdl_op_definition
class VarResultOp(Operation):
    name: str = "test.var_result_op"

    res: Annotated[list[OpResult], VarResultDef(StringAttr)]


def test_var_result_builder():
    op = VarResultOp.build(result_types=[[0, 1]])
    op.verify()
    assert [res.typ for res in op.results
            ] == [StringAttr.from_int(0),
                  StringAttr.from_int(1)]


@irdl_op_definition
class TwoVarResultOp(Operation):
    name: str = "test.two_var_result_op"

    res1: Annotated[list[OpResult], VarResultDef(StringAttr)]
    res2: Annotated[list[OpResult], VarResultDef(StringAttr)]
    irdl_options = [AttrSizedResultSegments()]


def test_two_var_result_builder():
    op = TwoVarResultOp.build(result_types=[[0, 1], [2, 3]])
    op.verify()
    assert [res.typ for res in op.results] == [
        StringAttr.from_int(0),
        StringAttr.from_int(1),
        StringAttr.from_int(2),
        StringAttr.from_int(3)
    ]

    dense_type = VectorType.from_type_and_list(IntegerType.from_width(32), [2])

    assert op.attributes[AttrSizedResultSegments.
                         attribute_name] == DenseIntOrFPElementsAttr.from_list(
                             dense_type, [2, 2])


def test_two_var_result_builder2():
    op = TwoVarResultOp.build(result_types=[[0], [1, 2, 3]])
    op.verify()
    assert [res.typ for res in op.results] == [
        StringAttr.from_int(0),
        StringAttr.from_int(1),
        StringAttr.from_int(2),
        StringAttr.from_int(3)
    ]
    dense_type = VectorType.from_type_and_list(IntegerType.from_width(32), [2])
    assert op.attributes[AttrSizedResultSegments.
                         attribute_name] == DenseIntOrFPElementsAttr.from_list(
                             dense_type, [1, 3])


@irdl_op_definition
class MixedResultOp(Operation):
    name: str = "test.mixed"

    res1: Annotated[list[OpResult], VarResultDef(StringAttr)]
    res2: Annotated[OpResult, ResultDef(StringAttr)]
    res3: Annotated[list[OpResult], VarResultDef(StringAttr)]
    irdl_options = [AttrSizedResultSegments()]


def test_var_mixed_builder():
    op = MixedResultOp.build(result_types=[[0, 1], 2, [3, 4]])
    op.verify()
    assert [res.typ for res in op.results] == [
        StringAttr.from_int(0),
        StringAttr.from_int(1),
        StringAttr.from_int(2),
        StringAttr.from_int(3),
        StringAttr.from_int(4)
    ]

    dense_type = VectorType.from_type_and_list(IntegerType.from_width(32), [3])

    assert op.attributes[AttrSizedResultSegments.
                         attribute_name] == DenseIntOrFPElementsAttr.from_list(
                             dense_type, [2, 1, 2])


#   ___                                 _
#  / _ \ _ __   ___ _ __ __ _ _ __   __| |
# | | | | '_ \ / _ \ '__/ _` | '_ \ / _` |
# | |_| | |_) |  __/ | | (_| | | | | (_| |
#  \___/| .__/ \___|_|  \__,_|_| |_|\__,_|
#       |_|
#


@irdl_op_definition
class OperandOp(Operation):
    name: str = "test.operand_op"

    res: Annotated[SSAValue, OperandDef(StringAttr)]


def test_operand_builder_operation():
    op1 = ResultOp.build(result_types=[0])
    op2 = OperandOp.build(operands=[op1])
    op2.verify()
    assert op2.operands == [op1.res]


def test_operand_builder_value():
    op1 = ResultOp.build(result_types=[0])
    op2 = OperandOp.build(operands=[op1.res])
    op2.verify()
    assert op2.operands == [op1.res]


def test_operand_builder_exception():
    with pytest.raises(ValueError):
        OperandOp.build()


@irdl_op_definition
class OptOperandOp(Operation):
    name: str = "test.opt_operand_op"

    res = OptOperandDef(StringAttr)


def test_opt_operand_builder():
    op = ResultOp.build(result_types=[0])
    op1 = OptOperandOp.build(operands=[[op]])
    op2 = OptOperandOp.build(operands=[[]])
    op1.verify()
    op2.verify()
    assert [operand for operand in op1.operands] == [op.res]
    assert len(op2.operands) == 0


def test_opt_operand_builder_two_args():
    op = ResultOp.build(result_types=[0])
    with pytest.raises(ValueError) as _:
        OptOperandOp.build(operands=[[op, op]])


@irdl_op_definition
class VarOperandOp(Operation):
    name: str = "test.var_operand_op"

    res = VarOperandDef(StringAttr)


def test_var_operand_builder():
    op1 = ResultOp.build(result_types=[0])
    op2 = VarOperandOp.build(operands=[[op1, op1]])
    op2.verify()
    assert op2.operands == [op1.res, op1.res]


@irdl_op_definition
class TwoVarOperandOp(Operation):
    name: str = "test.two_var_operand_op"

    res1 = VarOperandDef(StringAttr)
    res2 = VarOperandDef(StringAttr)
    irdl_options = [AttrSizedOperandSegments()]


def test_two_var_operand_builder():
    op1 = ResultOp.build(result_types=[0])
    op2 = TwoVarOperandOp.build(operands=[[op1, op1], [op1, op1]])
    op2.verify()
    assert op2.operands == [op1.res] * 4
    dense_type = VectorType.from_type_and_list(IntegerType.from_width(32), [2])
    assert op2.attributes[
        AttrSizedOperandSegments.
        attribute_name] == DenseIntOrFPElementsAttr.from_list(
            dense_type, [2, 2])


def test_two_var_operand_builder2():
    op1 = ResultOp.build(result_types=[0])
    op2 = TwoVarOperandOp.build(operands=[[op1], [op1, op1, op1]])
    op2.verify()
    assert op2.operands == [op1.res] * 4
    dense_type = VectorType.from_type_and_list(IntegerType.from_width(32), [2])
    assert op2.attributes[
        AttrSizedOperandSegments.
        attribute_name] == DenseIntOrFPElementsAttr.from_list(
            dense_type, [1, 3])


#      _   _   _        _ _           _
#     / \ | |_| |_ _ __(_) |__  _   _| |_ ___
#    / _ \| __| __| '__| | '_ \| | | | __/ _ \
#   / ___ \ |_| |_| |  | | |_) | |_| | ||  __/
#  /_/   \_\__|\__|_|  |_|_.__/ \__,_|\__\___|
#


@irdl_op_definition
class AttrOp(Operation):
    name: str = "test.two_var_result_op"
    attr = AttributeDef(StringAttr)


def test_attr_op():
    op = AttrOp.build(attributes={"attr": 0})
    op.verify()
    assert op.attr == StringAttr.from_int(0)


def test_attr_new_attr_op():
    op = AttrOp.build(attributes={
        "attr": 0,
        "new_attr": StringAttr.from_int(1)
    })
    op.verify()
    assert op.attr == StringAttr.from_int(0)
    assert op.attributes["new_attr"] == StringAttr.from_int(1)


@irdl_op_definition
class OptionalAttrOp(Operation):
    name: str = "test.opt_attr_op"

    opt_attr = OptAttributeDef(StringAttr)


def test_optional_attr_op_empty():
    op = OptionalAttrOp.build()
    op.verify()
    assert op.opt_attr is None


#  ____            _
# |  _ \ ___  __ _(_) ___  _ __
# | |_) / _ \/ _` | |/ _ \| '_ \
# |  _ <  __/ (_| | | (_) | | | |
# |_| \_\___|\__, |_|\___/|_| |_|
#            |___/
#


@irdl_op_definition
class RegionOp(Operation):
    name: str = "test.region_op"

    region = RegionDef()


def test_region_op_region():
    op = RegionOp.build(regions=[Region()])
    op.verify()
    assert op.region.blocks == []


def test_region_op_blocks():
    op = RegionOp.build(regions=[[Block(), Block()]])
    op.verify()
    assert len(op.region.blocks) == 2


def test_region_op_ops():
    op1 = RegionOp.build(regions=[[Block()]])
    op2 = RegionOp.build(regions=[[Block()]])
    op = RegionOp.build(regions=[[op1, op2]])
    op.verify()
    assert len(op.region.blocks) == 1
    assert len(op.region.blocks[0].ops) == 2


@irdl_op_definition
class SBRegionOp(Operation):
    name: str = "test.sbregion_op"

    region = SingleBlockRegionDef()


def test_sbregion_one_block():
    op = SBRegionOp.build(regions=[[Block()]])
    op.verify()
    assert len(op.region.blocks) == 1


@irdl_op_definition
class OptRegionOp(Operation):
    name: str = "test.opt_region_op"

    reg = OptRegionDef()


def test_opt_region_builder():
    op1 = OptRegionOp.build(regions=[[[Block(), Block()]]])
    op2 = OptRegionOp.build(regions=[[Region()]])
    op1.verify()
    op2.verify()


def test_opt_region_builder_two_args():
    with pytest.raises(ValueError) as _:
        OptRegionOp.build(regions=[[Region(), Region()]])


@irdl_op_definition
class OptSBRegionOp(Operation):
    name: str = "test.sbregion_op"

    region = OptSingleBlockRegionDef()


def test_opt_sbregion_one_block():
    op1 = OptSBRegionOp.build(regions=[[[Block()]]])
    op2 = OptSBRegionOp.build(regions=[[]])
    op1.verify()
    op2.verify()
    assert len(op1.region.blocks) == 1
    assert op2.region is None


@irdl_op_definition
class VarRegionOp(Operation):
    name: str = "test.var_operand_op"

    regs = VarRegionDef()


def test_var_region_builder():
    op = VarRegionOp.build(regions=[[Region(), [Block(), Block()]]])
    op.verify()
    assert len(op.regs[0].blocks) == 0  # type: ignore
    assert len(op.regs[1].blocks) == 2  # type: ignore


@irdl_op_definition
class VarSBRegionOp(Operation):
    name: str = "test.sbregion_op"

    regs = VarSingleBlockRegionDef()


def test_var_sbregion_one_block():
    op1 = VarSBRegionOp.build(regions=[[[Block()]]])
    op2 = VarSBRegionOp.build(regions=[[Region(), [Block(), Block()]]])
    op1.verify()
    op2.verify()
    assert len(op1.regs) == 1  # type: ignore
    assert len(op2.regs) == 2  # type: ignore
    assert len(op2.regs[0].blocks) == 0  # type: ignore
    assert len(op2.regs[1].blocks) == 2  # type: ignore


#  __  __ _
# |  \/  (_)___  ___
# | |\/| | / __|/ __|
# | |  | | \__ \ (__
# |_|  |_|_|___/\___|
#


def test_parent_pointers():
    op = ResultOp.build(result_types=[0])
    block = Block.from_ops([op])
    reg = Region.from_block_list([block])
    reg_op = RegionOp.build(regions=[reg])

    block_2 = Block.from_ops([reg_op])
    reg_2 = Region.from_block_list([block_2])
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
