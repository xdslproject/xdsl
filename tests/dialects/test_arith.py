import pytest

from xdsl.dialects.arith import (Addi, Constant, DivUI, DivSI, Subi,
                                 FloorDivSI, CeilDivSI, CeilDivUI, RemUI,
                                 RemSI, MinUI, MinSI, MaxUI, MaxSI, AndI, OrI,
                                 XOrI, ShLI, ShRUI, ShRSI, Cmpi, Addf, Subf,
                                 Mulf, Divf, Maxf, Minf)
from xdsl.dialects.scf import If
from xdsl.dialects.builtin import i32, f32


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
        condition = Cmpi.from_mnemonic(self.a, self.b, input)
        assert condition.operands[0].op is self.a
        assert condition.operands[1].op is self.b

        # op = If.get(condition)

        # import pdb;pdb.set_trace()


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
