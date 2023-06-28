from xdsl.backend.riscv.lowering.rv32_arith_lowering import LowerArithRV32
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.memref import MemRef
from xdsl.ir import MLContext
from xdsl.parser import Parser as IRParser

import pytest


def test_lower_arith():
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)
    ctx.register_dialect(MemRef)

    ctx.register_dialect(Func)

    samples = [
        "xdsl/backend/riscv/lowering/tests/examples/dot_product.mlir",
    ]

    for path in samples:
        with open(path, "r") as f:
            parser = IRParser(ctx, f.read(), name=f"{path}")
            module_op = parser.parse_module()
            LowerArithRV32().apply(ctx, module_op)
            with pytest.raises(Exception):
                module_op.verify()

    assert True
