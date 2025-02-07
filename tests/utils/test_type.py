from xdsl.dialects import test
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    MemRefType,
    NoneAttr,
    TensorType,
    UnrankedTensorType,
)
from xdsl.utils.type import (
    get_element_type_or_self,
    get_encoding,
    have_compatible_shape,
)


def test_get_element_type_or_self():
    scalar_type1 = test.TestType("foo")
    assert scalar_type1 == get_element_type_or_self(scalar_type1)

    shaped_type1 = TensorType(scalar_type1, [4])
    assert scalar_type1 == get_element_type_or_self(shaped_type1)

    unranked_shaped_type1 = UnrankedTensorType(scalar_type1)
    assert scalar_type1 == get_element_type_or_self(unranked_shaped_type1)


def test_get_encoding():
    scalar_type1 = test.TestType("foo")

    assert get_encoding(scalar_type1) == NoneAttr()

    shaped_type1 = TensorType(scalar_type1, [4])

    assert get_encoding(shaped_type1) == NoneAttr()

    shaped_type2 = TensorType(scalar_type1, [4], scalar_type1)

    assert get_encoding(shaped_type2) == scalar_type1

    shaped_type3 = MemRefType(scalar_type1, [4])

    assert get_encoding(shaped_type3) == NoneAttr()


def test_have_compatible_shape():
    scalar_type1 = test.TestType("foo")
    scalar_type2 = test.TestType("foo")

    assert have_compatible_shape(scalar_type1, scalar_type2)

    shaped_type1 = TensorType(scalar_type1, [4])

    assert not have_compatible_shape(scalar_type1, shaped_type1)
    assert not have_compatible_shape(shaped_type1, scalar_type1)

    unranked_shaped_type1 = UnrankedTensorType(scalar_type1)
    unranked_shaped_type2 = UnrankedTensorType(scalar_type2)

    assert have_compatible_shape(shaped_type1, unranked_shaped_type1)
    assert have_compatible_shape(unranked_shaped_type1, shaped_type1)
    assert have_compatible_shape(unranked_shaped_type1, unranked_shaped_type2)

    shaped_type2 = TensorType(scalar_type2, [1, 2])

    assert not have_compatible_shape(shaped_type1, shaped_type2)

    shaped_type3 = TensorType(scalar_type2, [5])

    assert not have_compatible_shape(shaped_type1, shaped_type3)

    shaped_type4 = TensorType(scalar_type2, [1, 3])

    assert not have_compatible_shape(shaped_type2, shaped_type4)

    shaped_type5 = TensorType(scalar_type2, [DYNAMIC_INDEX, 3])
    shaped_type6 = TensorType(scalar_type2, [1, DYNAMIC_INDEX])
    shaped_type7 = TensorType(scalar_type2, [DYNAMIC_INDEX, DYNAMIC_INDEX])

    assert have_compatible_shape(shaped_type4, shaped_type5)
    assert have_compatible_shape(shaped_type4, shaped_type6)

    assert have_compatible_shape(shaped_type5, shaped_type6)
    assert have_compatible_shape(shaped_type5, shaped_type7)

    assert have_compatible_shape(shaped_type6, shaped_type7)

    shaped_type8 = TensorType(scalar_type2, [2, DYNAMIC_INDEX])

    assert not have_compatible_shape(shaped_type4, shaped_type8)
