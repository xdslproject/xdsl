from typing import TypeVar

import pytest

from xdsl.dialects.arith import BinaryOperation, Constant
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    VectorType,
    f32,
    i32,
)
from xdsl.dialects.experimental.math import (
    AbsFOp,
    AbsIOp,
    Atan2Op,
    AtanOp,
    CbrtOp,
    CeilOp,
    CopySignOp,
    CosOp,
    CountLeadingZerosOp,
    CountTrailingZerosOp,
    CtPopOp,
    ErfOp,
    Exp2Op,
    ExpM1Op,
    ExpOp,
    FloorOp,
    FmaOp,
    FPowIOp,
    IPowIOp,
    Log1pOp,
    Log2Op,
    Log10Op,
    LogOp,
    PowFOp,
    RoundEvenOp,
    RoundOp,
    RsqrtOp,
    SinOp,
    SqrtOp,
    TanhOp,
    TanOp,
    TruncOp,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue

_BinOpArgT = TypeVar("_BinOpArgT", bound=Attribute)


class Test_float_math_binary_construction:
    operand_type = f32
    a = Constant(FloatAttr(1.1, operand_type))
    b = Constant(FloatAttr(2.2, operand_type))

    f32_vector_type = VectorType(f32, [3])

    lhs_vector_ssa_value = TestSSAValue(f32_vector_type)
    rhs_vector_ssa_value = TestSSAValue(f32_vector_type)

    lhs_vector = TestOp([lhs_vector_ssa_value], result_types=[f32_vector_type])
    rhs_vector = TestOp([rhs_vector_ssa_value], result_types=[f32_vector_type])

    f32_tensor_type = DenseIntOrFPElementsAttr.tensor_from_list([5.5], f32, [])
    lhs_tensor_ssa_value = TestSSAValue(f32_tensor_type)
    rhs_tensor_ssa_value = TestSSAValue(f32_tensor_type)

    lhs_tensor = TestOp([lhs_tensor_ssa_value], result_types=[f32_tensor_type])
    rhs_tensor = TestOp([rhs_tensor_ssa_value], result_types=[f32_tensor_type])

    @pytest.mark.parametrize(
        "OpClass",
        [
            Atan2Op,
            CopySignOp,
            IPowIOp,
            PowFOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_float_binary_ops_constant_math_init(
        self,
        OpClass: type[BinaryOperation[_BinOpArgT]],
        return_type: Attribute,
    ):
        op = OpClass(self.a, self.b)
        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == self.operand_type

    @pytest.mark.parametrize(
        "OpClass",
        [
            Atan2Op,
            CopySignOp,
            IPowIOp,
            PowFOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, f32_vector_type])
    def test_flaot_binary_vector_ops_init(
        self, OpClass: type[BinaryOperation[_BinOpArgT]], return_type: Attribute
    ):
        op = OpClass(self.lhs_vector, self.rhs_vector)
        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.lhs_vector
        assert op.rhs.owner is self.rhs_vector
        assert op.result.type == self.f32_vector_type

    @pytest.mark.parametrize(
        "OpClass",
        [
            Atan2Op,
            CopySignOp,
            IPowIOp,
            PowFOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, f32_tensor_type])
    def test_float_binary_ops_tensor_math_init(
        self, OpClass: type[BinaryOperation[_BinOpArgT]], return_type: Attribute
    ):
        op = OpClass(self.lhs_tensor, self.rhs_tensor)
        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.lhs_tensor
        assert op.rhs.owner is self.rhs_tensor
        assert op.result.type == self.f32_tensor_type


class Test_float_math_unary_constructions:
    operand_type = f32
    a = Constant(FloatAttr(1, operand_type))

    f32_vector_type = VectorType(f32, [3])

    test_vector_ssa = TestSSAValue(f32_vector_type)
    test_vec = TestOp([test_vector_ssa], result_types=[f32_vector_type])

    f32_tensor_type = DenseIntOrFPElementsAttr.tensor_from_list([5.5], f32, [])
    test_tensor_ssa_value = TestSSAValue(f32_tensor_type)
    test_tensor = TestOp([test_tensor_ssa_value], result_types=[f32_tensor_type])

    @pytest.mark.parametrize(
        "OpClass",
        [
            AbsFOp,
            AtanOp,
            CbrtOp,
            CeilOp,
            CosOp,
            ErfOp,
            Exp2Op,
            ExpM1Op,
            ExpOp,
            FloorOp,
            Log10Op,
            Log1pOp,
            Log2Op,
            LogOp,
            RoundEvenOp,
            RoundOp,
            RsqrtOp,
            SinOp,
            SqrtOp,
            TanOp,
            TanhOp,
            TruncOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [operand_type])
    def test_float_math_constant_ops_init(
        self,
        OpClass: type,
        return_type: Attribute,  # FIXME
    ):
        op = OpClass(self.a)
        assert op.result.type == f32
        assert op.operand.type == f32
        assert op.operand.owner == self.a
        assert op.result.type == self.operand_type

    @pytest.mark.parametrize(
        "OpClass",
        [
            AbsFOp,
            AtanOp,
            CbrtOp,
            CeilOp,
            CosOp,
            ErfOp,
            Exp2Op,
            ExpM1Op,
            ExpOp,
            FloorOp,
            Log10Op,
            Log1pOp,
            Log2Op,
            LogOp,
            RoundEvenOp,
            RoundOp,
            RsqrtOp,
            SinOp,
            SqrtOp,
            TanOp,
            TanhOp,
            TruncOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [f32_tensor_type])
    def test_float_math_ops_vector_init(self, OpClass: type, return_type: Attribute):
        op = OpClass(self.test_vec)
        assert op.result.type == self.f32_vector_type
        assert op.operand.type == self.f32_vector_type
        assert op.operand.owner == self.test_vec
        assert op.result.type == self.f32_vector_type

    @pytest.mark.parametrize(
        "OpClass",
        [
            AbsFOp,
            AtanOp,
            CbrtOp,
            CeilOp,
            CosOp,
            ErfOp,
            Exp2Op,
            ExpM1Op,
            ExpOp,
            FloorOp,
            Log10Op,
            Log1pOp,
            Log2Op,
            LogOp,
            RoundEvenOp,
            RoundOp,
            RsqrtOp,
            SinOp,
            SqrtOp,
            TanOp,
            TanhOp,
            TruncOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [f32_tensor_type])
    def test_float_math_ops_tensor_init(self, OpClass: type, return_type: Attribute):
        op = OpClass(self.test_tensor)
        assert op.result.type == self.f32_tensor_type
        assert op.operand.type == self.f32_tensor_type
        assert op.operand.owner == self.test_tensor
        assert op.result.type == self.f32_tensor_type


class Test_fpowi:
    a = Constant(FloatAttr(2.2, f32))
    b = Constant.from_int_and_width(0, 32)

    f32_vector_type = VectorType(f32, [3])

    a_vector_ssa_value = TestSSAValue(f32_vector_type)
    a_vector = TestOp([a_vector_ssa_value], result_types=[f32_vector_type])

    f32_tensor_type = DenseIntOrFPElementsAttr.tensor_from_list([5.5], f32, [])
    a_tensor_ssa_value = TestSSAValue(f32_tensor_type)
    a_tensor = TestOp([a_tensor_ssa_value], result_types=[f32_tensor_type])

    def test_fpowi_constant_construction(self):
        op = FPowIOp(self.a, self.b)
        assert op.result.type == f32
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == f32

    def test_fpowi_vector_construction(self):
        op = FPowIOp(self.a_vector, self.b)
        assert op.result.type == self.f32_vector_type
        assert op.lhs.owner is self.a_vector
        assert op.rhs.owner is self.b
        assert op.result.type == self.f32_vector_type

    def test_fpowi_tensor_construction(self):
        op = FPowIOp(self.a_tensor, self.b)
        assert op.result.type == self.f32_tensor_type
        assert op.lhs.owner is self.a_tensor
        assert op.rhs.owner is self.b
        assert op.result.type == self.f32_tensor_type


class Test_fma:
    a = Constant(FloatAttr(1.1, f32))
    b = Constant(FloatAttr(2.2, f32))
    c = Constant(FloatAttr(3.3, f32))

    f32_vector_type = VectorType(f32, [3])
    test_vector_ssa = TestSSAValue(f32_vector_type)
    a_vec = TestOp([test_vector_ssa], result_types=[f32_vector_type])
    b_vec = TestOp([test_vector_ssa], result_types=[f32_vector_type])
    c_vec = TestOp([test_vector_ssa], result_types=[f32_vector_type])

    f32_tensor_type = DenseIntOrFPElementsAttr.tensor_from_list([5.5], f32, [])
    test_tensor_ssa = TestSSAValue(f32_vector_type)
    a_tensor = TestOp([test_tensor_ssa], result_types=[f32_tensor_type])
    b_tensor = TestOp([test_tensor_ssa], result_types=[f32_tensor_type])
    c_tensor = TestOp([test_tensor_ssa], result_types=[f32_tensor_type])

    def test_fma_construction(self):
        op = FmaOp(self.a, self.b, self.c)
        assert op.result.type == f32
        assert op.a.owner is self.a
        assert op.b.owner is self.b
        assert op.c.owner is self.c
        assert op.result.type == f32

    def test_fma_construction_vector(self):
        op = FmaOp(self.a_vec, self.b_vec, self.c_vec)
        assert op.result.type == self.f32_vector_type
        assert op.a.owner is self.a_vec
        assert op.b.owner is self.b_vec
        assert op.c.owner is self.c_vec

    def test_fma_construction_tensor(self):
        op = FmaOp(self.a_tensor, self.b_tensor, self.c_tensor)
        assert op.result.type == self.f32_tensor_type
        assert op.a.owner is self.a_tensor
        assert op.b.owner is self.b_tensor
        assert op.c.owner is self.c_tensor


class Test_int_math_unary_constructions:
    operand_type = i32
    a = Constant.from_int_and_width(0, 32)

    i32_vector_type = VectorType(i32, [1])
    test_vector_ssa = TestSSAValue(i32_vector_type)
    test_vec = TestOp([test_vector_ssa], result_types=[i32_vector_type])

    i32_tensor_type = DenseIntOrFPElementsAttr.tensor_from_list([5], i32, [])
    test_tensor_ssa_value = TestSSAValue(i32_tensor_type)
    test_tensor = TestOp([test_tensor_ssa_value], result_types=[i32_tensor_type])

    @pytest.mark.parametrize(
        "OpClass",
        [
            AbsIOp,
            CountLeadingZerosOp,
            CountTrailingZerosOp,
            CtPopOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [operand_type])
    def test_int_math_ops_init(
        self,
        OpClass: type,
        return_type: Attribute,  # FIXME use something other than `type`
    ):
        op = OpClass(self.a)
        assert op.result.type == i32
        assert op.operand.type == i32
        assert op.operand.owner == self.a
        assert op.result.type == self.operand_type

    @pytest.mark.parametrize(
        "OpClass",
        [
            AbsIOp,
            CountLeadingZerosOp,
            CountTrailingZerosOp,
            CtPopOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [i32_vector_type])
    def test_int_math_ops_vec_init(
        self,
        OpClass: type,
        return_type: Attribute,  # FIXME use something other than `type`
    ):
        op = OpClass(self.test_vec)
        assert op.result.type == self.i32_vector_type
        assert op.operand.type == self.i32_vector_type
        assert op.operand.owner == self.test_vec
        assert op.result.type == self.i32_vector_type

    @pytest.mark.parametrize(
        "OpClass",
        [
            AbsIOp,
            CountLeadingZerosOp,
            CountTrailingZerosOp,
            CtPopOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [i32_vector_type])
    def test_int_math_ops_tensor_init(
        self,
        OpClass: type,
        return_type: Attribute,
    ):
        op = OpClass(self.test_tensor)
        assert op.result.type == self.i32_tensor_type
        assert op.operand.type == self.i32_tensor_type
        assert op.operand.owner == self.test_tensor
        assert op.result.type == self.i32_tensor_type


class Test_Trunci:
    operand_type = i32
    a = Constant.from_int_and_width(0, 32)

    i32_vector_type = VectorType(i32, [1])
    test_vector_ssa = TestSSAValue(i32_vector_type)
    test_vec = TestOp([test_vector_ssa], result_types=[i32_vector_type])

    i32_tensor_type = DenseIntOrFPElementsAttr.tensor_from_list([5], i32, [])
    test_tensor_ssa_value = TestSSAValue(i32_tensor_type)
    test_tensor = TestOp([test_tensor_ssa_value], result_types=[i32_tensor_type])

    def test_trunci_incorrect_bitwidth(self):
        with pytest.raises(VerifyException):
            _trunci_op = TruncOp(self.a).verify()
        with pytest.raises(VerifyException):
            _trunci_op_vec = TruncOp(self.test_vec).verify()
        with pytest.raises(VerifyException):
            _trunci_op_tensor = TruncOp(self.test_tensor).verify()
