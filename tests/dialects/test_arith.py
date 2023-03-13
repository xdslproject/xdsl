import pytest

from xdsl.dialects.arith import (Addi, Constant, DivUI, DivSI, Subi,
                                 FloorDivSI, CeilDivSI, CeilDivUI, RemUI,
                                 RemSI, MinUI, MinSI, MaxUI, MaxSI, AndI, OrI,
                                 XOrI, ShLI, ShRUI, ShRSI, Cmpi, Addf, Subf,
                                 Mulf, Divf, Maxf, Minf, IndexCastOp, FPToSIOp,
                                 SIToFPOp)
from xdsl.dialects.builtin import i32, f32, IndexType, IntegerType, Float32Type


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
