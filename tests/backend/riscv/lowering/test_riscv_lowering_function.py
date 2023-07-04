from xdsl.backend.riscv.lowering.riscv_arith_lowering import RISCVLowerArith
from xdsl.backend.riscv.lowering.riscv_stack_memref_lowering import (
    RISCVStackMemrefLower,
)
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.memref import MemRef
from xdsl.ir import MLContext
from xdsl.parser import Parser as IRParser

import pytest

from xdsl.printer import Printer


def test_lower_memref():
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)
    ctx.register_dialect(MemRef)

    ctx.register_dialect(Func)

    samples = [
        "tests/backend/riscv/lowering/dot_product.mlir",
    ]

    printer = Printer()

    for path in samples:
        with open(path, "r") as f:
            parser = IRParser(ctx, f.read(), name=f"{path}")
            module_op = parser.parse_module()
            RISCVStackMemrefLower().apply(ctx, module_op)
            RISCVLowerArith().apply(ctx, module_op)
            printer.print(module_op)
            module_op.verify()
