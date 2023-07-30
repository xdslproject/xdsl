from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32
from xdsl.ir.core import OpResult
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.rewriting.query import *
from xdsl.rewriting.sasha_rewrite_pattern import *
from xdsl.utils.hints import isa


@rewrite_pattern_query
def add_zero_query(root: arith.Addi, rhs_input: arith.Constant):
    return (
        isa(root.rhs, OpResult)
        and root.rhs.op == rhs_input
        and rhs_input.value == IntegerAttr.from_int_and_width(0, 32)
    )


def test_query_builder():
    assert list(add_zero_query.variables) == ["root", "0", "1", "rhs_input", "2"]

    (
        root_var,
        root_rhs,
        root_rhs_op,
        rhs_input,
        rhs_input_value,
    ) = add_zero_query.variables.values()

    assert isinstance(root_var, OperationVariable)
    assert isinstance(root_rhs, SSAValueVariable)
    assert isinstance(root_rhs_op, OperationVariable)
    assert isinstance(rhs_input, OperationVariable)
    assert isinstance(rhs_input_value, AttributeVariable)

    assert add_zero_query.constraints == [
        TypeConstraint(root_var, arith.Addi),
        OperationOperandConstraint(root_var, "rhs", root_rhs),
        TypeConstraint(root_rhs, OpResult),
        OpResultOpConstraint(root_rhs, root_rhs_op),
        EqConstraint(root_rhs_op, rhs_input),
        TypeConstraint(rhs_input, arith.Constant),
        OperationAttributeConstraint(rhs_input, "value", rhs_input_value),
        AttributeValueConstraint(
            rhs_input_value, IntegerAttr.from_int_and_width(0, 32)
        ),
    ]


@query_rewrite_pattern(add_zero_query)
def add_zero(rewriter: PatternRewriter, root: arith.Addi, rhs_input: arith.Constant):
    rewriter.replace_matched_op((), (root.lhs,))


def test_add_zero_rewrite():
    @ModuleOp
    @Builder.implicit_region
    def module():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            c0 = arith.Constant.from_int_and_width(0, i32).result
            c1 = arith.Constant.from_int_and_width(1, i32).result
            s = arith.Addi(c1, c0).result
            func.Call("func", ((s,)), ())
            func.Return()

    @ModuleOp
    @Builder.implicit_region
    def expected():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            _c0 = arith.Constant.from_int_and_width(0, i32).result
            c1 = arith.Constant.from_int_and_width(1, i32).result
            func.Call("func", ((c1,)), ())
            func.Return()

    PatternRewriteWalker(add_zero).rewrite_module(module)

    assert str(module) == str(expected)
