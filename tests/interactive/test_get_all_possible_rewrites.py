from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    ModuleOp,
    StringAttr,
)
from xdsl.dialects.test import TestOp
from xdsl.interactive.get_all_possible_rewrites import (
    IndexedIndividualRewrite,
    IndividualRewrite,
    get_all_possible_rewrites,
)
from xdsl.ir import Block, MLContext, Region
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms import individual_rewrite


def test_get_all_possible_rewrite():
    # build module
    prog = """
    builtin.module {
    "test.op"() {"label" = "a"}
    "test.op"() {"label" = "a"}
    "test.op"() {"label" = "b"}
    }
    """

    ctx = MLContext(True)
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    parser = Parser(ctx, prog)
    module = parser.parse_module()

    class Rewrite(RewritePattern):
        label: str

        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: TestOp, rewriter: PatternRewriter):
            if op.attributes["label"] != StringAttr(self.label):
                return
            rewriter.replace_matched_op(TestOp(attributes={"label": StringAttr("c")}))

    expected_res = (
        (
            IndexedIndividualRewrite(
                3, IndividualRewrite(operation="arith.addi", pattern="AddImmediateZero")
            )
        ),
    )

    res = get_all_possible_rewrites(module, individual_rewrite.REWRITE_BY_NAMES)
    assert res == expected_res


def test_empty_get_all_possible_rewrite():
    # build module
    index = IndexType()
    module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(module.body):
        function = func.FuncOp("hello", ((index,), (index,)))
        with ImplicitBuilder(function.body) as (n,):
            two = arith.Constant(IntegerAttr(2, index)).result
            three = arith.Constant(IntegerAttr(2, index)).result
            res_1 = arith.Muli(n, two)
            res_2 = arith.Muli(n, three)
            res = arith.Muli(res_1, res_2)
            func.Return(res)

    expected_res = ()

    res = get_all_possible_rewrites(module, individual_rewrite.REWRITE_BY_NAMES)
    assert res == expected_res
