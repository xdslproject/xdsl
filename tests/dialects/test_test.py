from xdsl.dialects.test import ProduceValuesOp, StrType


def test_produce_values_from_result_types():
    type1 = StrType("ga")
    type2 = StrType("bu")
    op = ProduceValuesOp.from_result_types(type1, type2)

    assert len(op.results) == 2
    assert op.results[0].typ == type1
    assert op.results[1].typ == type2


def test_produce_values_get_values():
    type1 = StrType("ga")
    type2 = StrType("bu")
    op, (val1, val2) = ProduceValuesOp.get_values(type1, type2)

    assert len(op.results) == 2
    assert op.results[0] == val1
    assert op.results[1] == val2
    assert val1.typ == type1
    assert val2.typ == type2
