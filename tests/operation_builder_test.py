from __future__ import annotations

import typing

from xdsl.ir import ParametrizedAttribute, Data, Block
from xdsl.irdl import *
import pytest
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, VectorType, IntegerType


@irdl_attr_definition
class StringAttr(Data):
    name = "test.string_attr"
    param: str

    @staticmethod
    @builder
    def from_int(i: int) -> StringAttr:
        return StringAttr(str(i))

    @staticmethod
    def parse(parser: Parser) -> Data:
        pass

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class ResultOp(Operation):
    name: str = "test.result_op"

    res = ResultDef(StringAttr)


def test_result_builder():
    op = ResultOp.build(result_types=[0])
    op.verify()
    assert [res.typ for res in op.results] == [StringAttr.from_int(0)]


def test_result_builder_exception():
    with pytest.raises(ValueError) as e:
        ResultOp.build()


@irdl_op_definition
class VarResultOp(Operation):
    name: str = "test.var_result_op"

    res = VarResultDef(StringAttr)


def test_var_result_builder():
    op = VarResultOp.build(result_types=[[0, 1]])
    op.verify()
    assert [res.typ for res in op.results
            ] == [StringAttr.from_int(0),
                  StringAttr.from_int(1)]


@irdl_op_definition
class TwoVarResultOp(Operation):
    name: str = "test.two_var_result_op"

    res1 = VarResultDef(StringAttr)
    res2 = VarResultDef(StringAttr)
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
class OperandOp(Operation):
    name: str = "test.operand_op"

    res = OperandDef(StringAttr)


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
    with pytest.raises(ValueError) as e:
        OperandOp.build()


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
class RegionOp(Operation):
    name: str = "test.two_var_result_op"

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
