from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr


def test_data_get():
    s = "Hello world"
    s_attr = StringAttr(s)
    assert StringAttr.get(s) == s_attr
    assert StringAttr.get(s_attr) == s_attr

    i = 42
    i_attr = IntAttr(i)
    assert IntAttr.get(i) == i_attr
    assert IntAttr.get(i_attr) == i_attr

    a = (
        IntAttr(1),
        IntAttr(2),
        IntAttr(3),
    )
    a_attr = ArrayAttr(a)
    assert ArrayAttr.get(a) == a_attr
    assert ArrayAttr.get(a_attr) == a_attr
