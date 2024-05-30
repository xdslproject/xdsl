from xdsl.dialects import builtin, test
from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt import layout_graph



def test_embed_into_abstract_layout():
    child_1 = dlt.MemberLayoutAttr(dlt.PrimitiveLayoutAttr(builtin.f32), "struct", "1")
    child_2 = dlt.MemberLayoutAttr(dlt.PrimitiveLayoutAttr(builtin.f32), "struct", "2")
    parent_layout = dlt.AbstractLayoutAttr([dlt.AbstractChildAttr([],[],child_1), dlt.AbstractChildAttr([],[],child_2)])
    new_layout = dlt.StructLayoutAttr([child_1, child_2])
    output = layout_graph.embed_layout_in(new_layout, parent_layout, set(), set(), set())
    assert output is not None



