import pytest

from xdsl.dialects.arith import (Addi, Constant, DivUI, DivSI, Subi,
                                 FloorDivSI, CeilDivSI, CeilDivUI, RemUI,
                                 RemSI, MinUI, MinSI, MaxUI, MaxSI, AndI, OrI,
                                 XOrI, ShLI, ShRUI, ShRSI, Cmpi, Addf, Subf,
                                 Mulf, Divf, Maxf, Minf, IndexCastOp, FPToSIOp,
                                 SIToFPOp, ExtFOp, TruncFOp)
from xdsl.dialects.builtin import i32, f32, f64, IndexType, IntegerType, Float32Type, FloatAttr


class Test_integer_arith_construction:
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(1, i32)

    @pytest.mark.parametrize(
        "func",
        [
            Addi, Subi, DivUI, DivSI, FloorDivSI, CeilDivSI, CeilDivUI, RemUI,
            RemSI, MinUI, MinSI, MaxUI, MaxSI, AndI, OrI, XOrI, ShLI, ShRUI,
            ShRSI
        ],
    )
    def test_arith_ops(self, func):
        op = func.get(self.a, self.b)
        assert op.operands[0].op is self.a
        assert op.operands[1].op is self.b

    def test_Cmpi(self):
        _ = Cmpi.get(self.a, self.b, 2)

    @pytest.mark.parametrize(
        "input",
        ["eq", "ne", "slt", "sle", "ult", "ule", "ugt", "uge"],
    )
    def test_Cmpi_from_mnemonic(self, input):
        _ = Cmpi.from_mnemonic(self.a, self.b, input)


class Test_float_arith_construction:

    a = Constant.from_float_and_width(1.1, f32)
    b = Constant.from_float_and_width(2.2, f32)

    @pytest.mark.parametrize(
        "func",
        [Addf, Subf, Mulf, Divf, Maxf, Minf],
    )
    def test_arith_ops(self, func):
        op = func.get(self.a, self.b)
        assert op.operands[0].op is self.a
        assert op.operands[1].op is self.b


def test_float_arith_construction_int_width():
    a = Constant.from_float_and_width(1.1, 32)
    b = Constant.from_float_and_width(1.1, 64)

    assert a.result.typ == f32
    assert a.value.type == f32
    assert b.result.typ == f64
    assert b.value.type == f64


def test_float_arith_construction_int_width_illegal():
    try:
        a = Constant.from_float_and_width(1.1, 48)
        # This should raise ValueError as 48 is an illegal float width
        assert False
    except ValueError:
        pass


def test_float_arith_construction_int_width_missmatch():
    try:
        a = Constant.from_float_and_width(FloatAttr(1.4, 64), 32)
        # This should raise TypeError as the type width in FloatAttr is
        # mismatched against the integer width provided
        assert False
    except TypeError:
        pass

    try:
        a = Constant.from_float_and_width(FloatAttr(1.4, 64), i32)
        # This should raise TypeError as the type width in FloatAttr is
        # mismatched against the float type provided
        assert False
    except TypeError:
        pass


def test_index_cast_op():
    a = Constant.from_int_and_width(0, 32)
    cast = IndexCastOp.get(a, IndexType())

    assert cast.result.typ == IndexType()
    assert cast.input.typ == i32
    assert cast.input.op == a


def test_cast_fp_and_si_ops():
    a = Constant.from_int_and_width(0, 32)
    fp = SIToFPOp.get(a, f32)
    si = FPToSIOp.get(fp, i32)

    assert fp.input == a.result
    assert fp.result == si.input
    assert isinstance(si.result.typ, IntegerType)
    assert fp.result.typ == f32


def test_extend_truncate_fpops():
    a = Constant.from_float_and_width(1.0, f32)
    b = Constant.from_float_and_width(2.0, f64)
    ext_op = ExtFOp.get(a, f64)
    trunc_op = TruncFOp.get(b, f32)

    assert ext_op.input == a.result
    assert ext_op.result.typ == f64
    assert trunc_op.input == b.result
    assert trunc_op.result.typ == f32
