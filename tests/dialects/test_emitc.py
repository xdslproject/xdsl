import pytest

from xdsl.dialects.builtin import DYNAMIC_INDEX, i1, i32
from xdsl.dialects.emitc import (
    EmitC_AddOp,
    EmitC_ArrayType,
    EmitC_BinaryOperation,
    EmitC_BitwiseAndOp,
    EmitC_BitwiseLeftShiftOp,
    EmitC_BitwiseNotOp,
    EmitC_BitwiseOrOp,
    EmitC_BitwiseRightShiftOp,
    EmitC_BitwiseXorOp,
    EmitC_CallOpaqueOp,
    EmitC_ConstantOp,
    EmitC_DivOp,
    EmitC_LogicalAndOp,
    EmitC_LogicalNotOp,
    EmitC_LogicalOrOp,
    EmitC_MulOp,
    EmitC_RemOp,
    EmitC_SubOp,
    EmitC_UnaryOperation,
    IntegerAttr,
)
from xdsl.ir import Attribute
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="expected static shape, but got dynamic dimension"
    ):
        EmitC_ArrayType([DYNAMIC_INDEX], i32)


def test_call_opaque_with_str_callee():
    """
    Test that EmitC_CallOpaqueOp can be created with a string callee.
    """
    EmitC_CallOpaqueOp(callee="test", call_args=[], result_types=[])


class Test_emitc_arith:
    operand_type = i32
    a = EmitC_ConstantOp(IntegerAttr(3, 32))
    b = EmitC_ConstantOp(IntegerAttr(4, 32))

    @pytest.mark.parametrize(
        "OpClass",
        [
            EmitC_AddOp,
            EmitC_SubOp,
            EmitC_MulOp,
            EmitC_DivOp,
            EmitC_RemOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_arith_ops_init(
        self,
        OpClass: type[EmitC_BinaryOperation],
        return_type: Attribute,
    ):
        op = OpClass(self.a.result, self.b.result, i32)

        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == self.operand_type


class Test_emitc_bitwise:
    operand_type = i32
    a = EmitC_ConstantOp(IntegerAttr(127, 32))
    b = EmitC_ConstantOp(IntegerAttr(12, 32))

    @pytest.mark.parametrize(
        "OpClass",
        [
            EmitC_BitwiseAndOp,
            EmitC_BitwiseLeftShiftOp,
            EmitC_BitwiseOrOp,
            EmitC_BitwiseRightShiftOp,
            EmitC_BitwiseXorOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_bitwise_ops_init(
        self,
        OpClass: type[EmitC_BinaryOperation],
        return_type: Attribute,
    ):
        op = OpClass(self.a.result, self.b.result, i32)

        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == self.operand_type

    @pytest.mark.parametrize(
        "NotOpClass",
        [
            EmitC_BitwiseNotOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_bitwise_not_op_init(
        self,
        NotOpClass: type[EmitC_UnaryOperation],
        return_type: Attribute,
    ):
        op = NotOpClass(self.a.result, i32)

        assert isinstance(op, NotOpClass)
        assert op.operand.owner is self.a
        assert op.result.type == self.operand_type


class Test_emitc_logical:
    operand_type = i1
    a = EmitC_ConstantOp(IntegerAttr(0, 1))
    b = EmitC_ConstantOp(IntegerAttr(1, 1))

    @pytest.mark.parametrize(
        "OpClass",
        [
            EmitC_LogicalAndOp,
            EmitC_LogicalOrOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_logical_ops_init(
        self,
        OpClass: type[EmitC_BinaryOperation],
        return_type: Attribute,
    ):
        op = OpClass(self.a.result, self.b.result, i1)

        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == self.operand_type

    @pytest.mark.parametrize(
        "NotOpClass",
        [
            EmitC_LogicalNotOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_logical_not_op_init(
        self,
        NotOpClass: type[EmitC_UnaryOperation],
        return_type: Attribute,
    ):
        op = NotOpClass(self.a.result, i1)

        assert isinstance(op, NotOpClass)
        assert op.operand.owner is self.a
        assert op.result.type == self.operand_type
