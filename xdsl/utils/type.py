"""
Type utilities.
"""

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ContainerType,
    NoneAttr,
    ShapedType,
    TensorType,
)
from xdsl.ir import Attribute
from xdsl.utils.hints import isa


def get_element_type_or_self(maybe_container_type: Attribute) -> Attribute:
    """
    If the input is a `ContainerType`, then returns it's element type, otherwise returns
    input.
    """
    if isa(maybe_container_type, ContainerType):
        return maybe_container_type.get_element_type()
    return maybe_container_type


def get_encoding(maybe_shaped_type: Attribute) -> Attribute:
    if isinstance(maybe_shaped_type, TensorType):
        return maybe_shaped_type.encoding
    return NoneAttr()


def have_compatible_shape(lhs_type: Attribute, rhs_type: Attribute) -> bool:
    is_lhs_container = isinstance(lhs_type, ContainerType)
    is_rhs_container = isinstance(rhs_type, ContainerType)

    # both are scalars
    if not is_lhs_container and not is_rhs_container:
        return True

    # one is scalar and the other shaped
    if is_lhs_container != is_rhs_container:
        return False

    # at least one is unranked
    if not isinstance(lhs_type, ShapedType) or not isinstance(rhs_type, ShapedType):
        return True

    # both ranked, so check ranks
    if lhs_type.get_num_dims() != rhs_type.get_num_dims():
        return False

    return all(
        dim1 == DYNAMIC_INDEX or dim2 == DYNAMIC_INDEX or dim1 == dim2
        for dim1, dim2 in zip(lhs_type.get_shape(), rhs_type.get_shape())
    )


def are_tosa_broadcastable(t_in1: Attribute, t_in2: Attribute, t_out: Attribute):
    if (
        not isinstance(t_in1, ShapedType)
        or not isinstance(t_in2, ShapedType)
        or not isinstance(t_out, ShapedType)
    ):
        return False

    # check ranks are equal
    if not (t_in1.get_num_dims() == t_in2.get_num_dims() == t_out.get_num_dims()):
        return False

    # check ranks are broadcastable
    in_shapes = zip(t_in1.get_shape(), t_in2.get_shape())

    if not all(dim1 == dim2 or dim1 == 1 or dim2 == 1 for dim1, dim2 in in_shapes):
        return False

    # check output shape is constructed from input shapes
    shapes = zip(t_in1.get_shape(), t_in2.get_shape(), t_out.get_shape())
    return all(dim_out == max(dim1, dim2) for dim1, dim2, dim_out in shapes)
