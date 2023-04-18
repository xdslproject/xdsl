from xdsl.dialects.builtin import AnyArrayAttr, StringAttr
from xdsl.dialects.irdl import DialectOp, OperationOp, OperandsOp, ResultsOp, TypeOp
from xdsl.ir import Block, Region


def test_dialect_accessors():
    """
    Create a dialect with some operations and types, and check that we can
    retrieve the list of operations, or the list of types.
    """
    type1 = TypeOp.create(attributes={"name": StringAttr("type1")},
                          regions=[Region()])
    type2 = TypeOp.create(attributes={"name": StringAttr("type2")},
                          regions=[Region()])
    op1 = OperationOp.create(attributes={"name": StringAttr("op1")},
                             regions=[Region()])
    op2 = OperationOp.create(attributes={"name": StringAttr("op2")},
                             regions=[Region()])
    dialect = DialectOp.create(
        attributes={"name": StringAttr("test")},
        regions=[Region([Block([type1, type2, op1, op2])])])

    assert dialect.get_op_defs() == [op1, op2]
    assert dialect.get_type_defs() == [type1, type2]


def test_operation_accessors():
    """
    Create an operation, and check that we can retrieve the operands and
    results definition.
    """

    operands = OperandsOp.create(attributes={"params": AnyArrayAttr([])})
    results = ResultsOp.create(attributes={"params": AnyArrayAttr([])})

    # Check it on an operation that has operands and results
    op = OperationOp.create(attributes={"name": StringAttr("op")},
                            regions=[Region([Block([operands, results])])])

    assert op.get_operands() is operands
    assert op.get_results() is results

    # Check it on an operation that has no operands and results
    empty_op = OperationOp.create(attributes={"name": StringAttr("op")},
                                  regions=[Region([Block()])])

    assert empty_op.get_operands() is None
    assert empty_op.get_results() is None
