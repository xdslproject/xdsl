from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir.core import OpResult
from xdsl.rewriting.query import *
from xdsl.rewriting.query_builder import PatternQuery
from xdsl.rewriting.query_rewrite_pattern import *
from xdsl.utils.hints import isa


@PatternQuery
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
        OperationOperandConstraint(root_var, root_rhs, "rhs"),
        TypeConstraint(root_rhs, OpResult),
        OpResultOpConstraint(root_rhs, root_rhs_op),
        EqConstraint(root_rhs_op, rhs_input),
        TypeConstraint(rhs_input, arith.Constant),
        OperationAttributeConstraint(rhs_input, rhs_input_value, "value"),
        AttributeValueConstraint(
            rhs_input_value, IntegerAttr.from_int_and_width(0, 32)
        ),
    ]
