from xdsl.dialects import builtin
from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt.layout_manipulation import Manipulator


def test_embed_into_abstract_layout():
    child_1 = dlt.MemberLayoutAttr(dlt.PrimitiveLayoutAttr(builtin.f32), "struct", "1")
    child_2 = dlt.MemberLayoutAttr(dlt.PrimitiveLayoutAttr(builtin.f32), "struct", "2")
    parent_layout = dlt.AbstractLayoutAttr([dlt.AbstractChildAttr([],[],child_1), dlt.AbstractChildAttr([],[],child_2)])
    new_layout = dlt.StructLayoutAttr([child_1, child_2])
    output = Manipulator.embed_layout_in(new_layout, parent_layout, set(), set(), set(), False)
    assert output is not None



