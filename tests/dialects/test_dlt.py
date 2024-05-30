from xdsl.dialects import builtin
from xdsl.dialects.experimental import dlt


def test_typetype_has_selectable_type():
    t1 = dlt.TypeType([
        dlt.ElementAttr([("a","1"), ("b","1")], [], builtin.f32),
        dlt.ElementAttr([("a", "1"), ("b","2")], [], builtin.f32),
        dlt.ElementAttr([("a", "1"), ("b","3")], [], builtin.f32)
    ])
    t2 =  dlt.TypeType([
        dlt.ElementAttr([("b","1")], [], builtin.f32),
        dlt.ElementAttr([("b","2")], [], builtin.f32),
    ])
    selections = t1.has_selectable_type(t2)
    assert selections == {(dlt.SetAttr([dlt.MemberAttr("a", "1")]), dlt.SetAttr([]))}