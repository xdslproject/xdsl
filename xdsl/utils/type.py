"""
Type utilities.
"""

from typing import Any, cast

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ContainerType,
    NoneAttr,
    ShapedType,
    TensorType,
)
from xdsl.ir import Attribute


def get_element_type_or_self(maybe_shaped_type: Attribute) -> Attribute:
    if isinstance(maybe_shaped_type, ContainerType):
        container_type = cast(ContainerType[Any], maybe_shaped_type)
        return container_type.get_element_type()
    return maybe_shaped_type


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
