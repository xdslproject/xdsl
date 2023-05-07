import pytest
from xdsl.transforms.lower_snitch import LowerSsrSetDimensionBoundOp
from xdsl.dialects import riscv, snitch, test, builtin
from xdsl.ir import MLContext
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.parser import Parser


def test_dimension_oob():
    prog = """
"builtin.module"() ({
  %stream = "test.op"() : () -> !riscv.reg<>
  %bound = "test.op"() : () -> !riscv.reg<>
  "snitch.ssr_set_dimension_bound"(%stream, %bound) {"dimension" = 4 : i32} : (!riscv.reg<>, !riscv.reg<>) -> ()
}) : () -> ()
"""

    ctx = MLContext()
    ctx.register_dialect(builtin.Builtin)
    ctx.register_dialect(test.Test)
    ctx.register_dialect(riscv.RISCV)
    ctx.register_dialect(snitch.Snitch)
    parser = Parser(ctx, prog)
    module = parser.parse_module()
    walker = PatternRewriteWalker(LowerSsrSetDimensionBoundOp())
    with pytest.raises(AssertionError):
        walker.rewrite_module(module)
