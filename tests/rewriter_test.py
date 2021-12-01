from xdsl.printer import Printer
from xdsl.dialects.builtin import Builtin, IntegerAttr
from xdsl.parser import Parser
from xdsl.dialects.std import Std, Constant
from xdsl.ir import MLContext
from xdsl.rewriter import *
from io import StringIO


def rewrite_and_compare(ctx: MLContext, prog: str, expected_prog: str,
                        walker: PatternRewriteWalker):
    parser = Parser(ctx, prog)
    module = parser.parse_op()

    walker.rewrite_module(module)
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected_prog.strip()


def test_simple_rewrite():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    prog = \
    """module() {
%0 : !i32 = std.constant() ["value" = 42 : !i32]
%1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 43 : !i64]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32) 
}"""

    class RewriteConst(RewritePattern):
        def match_and_rewrite(
                self, op: Operation,
                new_operands: List[SSAValue]) -> Optional[RewriteAction]:
            if isinstance(op, Constant) and op.value.value.data == 42:
                return RewriteAction.from_op_list(
                    [std.constant_from_attr(IntegerAttr.get(43), std.i32)])
            return None

    rewrite_and_compare(ctx, prog, expected,
                        PatternRewriteWalker(RewriteConst()))


if __name__ == "__main__":
    test_simple_rewrite()
