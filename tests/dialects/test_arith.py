import pytest

from xdsl.dialects.arith import (Addi, Constant, DivUI, DivSI, Subi,
                                 FloorDivSI, CeilDivSI, CeilDivUI, RemUI,
                                 RemSI, MinUI, MinSI, MaxUI, MaxSI, AndI, OrI,
                                 XOrI, ShLI, ShRUI, ShRSI, Cmpi, Addf, Subf,
                                 Mulf, Divf, Maxf, Minf, IndexCastOp, FPToSIOp,
                                 SIToFPOp, ExtFOp, TruncFOp, Cmpf)
from xdsl.dialects.builtin import i32, i64, f32, f64, IndexType, IntegerType, Float32Type


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
        _ = Cmpi.get(self.a, self.b, input)


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


def test_cmpf_from_mnemonic():
    a = Constant.from_float_and_width(1.0, f64)
    b = Constant.from_float_and_width(2.0, f64)
    cmpf_ops = [None] * 10

    cmpf_ops[0] = Cmpf.get(a, b, "eq")
    cmpf_ops[1] = Cmpf.get(a, b, "ne")
    cmpf_ops[2] = Cmpf.get(a, b, "slt")
    cmpf_ops[3] = Cmpf.get(a, b, "sle")
    cmpf_ops[4] = Cmpf.get(a, b, "sgt")
    cmpf_ops[5] = Cmpf.get(a, b, "sge")
    cmpf_ops[6] = Cmpf.get(a, b, "ult")
    cmpf_ops[7] = Cmpf.get(a, b, "ule")
    cmpf_ops[8] = Cmpf.get(a, b, "ugt")
    cmpf_ops[9] = Cmpf.get(a, b, "uge")

    for index, op in enumerate(cmpf_ops):
        assert op.lhs.typ == f64
        assert op.rhs.typ == f64
        assert op.predicate.value.data == index


def test_cmpf_get():
    a = Constant.from_float_and_width(1.0, f32)
    b = Constant.from_float_and_width(2.0, f32)

    cmpf_op = Cmpf.get(a, b, 1)

    assert cmpf_op.lhs.typ == f32
    assert cmpf_op.rhs.typ == f32
    assert cmpf_op.predicate.value.data == 1


def test_cmpf_missmatch_type():
    a = Constant.from_float_and_width(1.0, f32)
    b = Constant.from_float_and_width(2.0, f64)

    with pytest.raises(TypeError) as e:
        cmpf_op = Cmpf.get(a, b, 1)
    assert e.value.args[
        0] == "Comparison operands must have same type, but provided !f32 and !f64"


def test_cmpi_missmatch_type():
    a = Constant.from_float_and_width(1, i32)
    b = Constant.from_float_and_width(2, i64)

    with pytest.raises(TypeError) as e:
        cmpi_op = Cmpi.get(a, b, 1)
    assert e.value.args[
        0] == "Comparison operands must have same type, but provided !i32 and !i64"
