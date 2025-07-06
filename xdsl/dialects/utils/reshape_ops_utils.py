"""
Utilities used by reshape ops.
See [MLIR counterpart](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h)
for more details.
"""

from dataclasses import dataclass
from typing import cast

from typing_extensions import TypeVar

from xdsl.dialects.builtin import I64, Annotated, ArrayAttr, IntegerAttr, ShapedType
from xdsl.ir import Attribute
from xdsl.irdl import (
    AtLeast,
    AttrConstraint,
    ConstraintContext,
    GenericAttrConstraint,
    irdl_to_attr_constraint,
)
from xdsl.utils.exceptions import VerifyException

_CONTIGUOUS_ARRAY_TYPE_CONSTRAINT = irdl_to_attr_constraint(
    ArrayAttr[
        ArrayAttr[
            Annotated[
                IntegerAttr[I64],
                IntegerAttr.constr(value=AtLeast(0)),
            ]
        ]
    ]
)

ArrayOfIntArrayAttr = ArrayAttr[ArrayAttr[IntegerAttr]]


@dataclass(frozen=True)
class ContiguousArrayOfIntArray(GenericAttrConstraint[ArrayOfIntArrayAttr]):
    """
    Enforce an ArrayAttr of ArrayAttr[IntegerAttr] to contain contiguous integer values across all inner arrays.
    For example: [[0, 1], [2, 3]] is valid, but [[3, 4], [0, 1]] is not.
    An empty inner array is considered contiguous.
    """

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        _CONTIGUOUS_ARRAY_TYPE_CONSTRAINT.verify(
            attr, constraint_context=constraint_context
        )
        attr = cast(ArrayOfIntArrayAttr, attr)

        # Flatten all integer values from all inner arrays
        flat_values = [e.value.data for inner in attr.data for e in inner.data]
        # Check that the flattened list is contiguous
        for prev, curr in zip(flat_values, flat_values[1:]):
            if curr != prev + 1:
                raise VerifyException(f"All inner arrays must be contiguous: {attr}")

    def mapping_type_vars(
        self, type_var_mapping: dict[TypeVar, AttrConstraint]
    ) -> "ContiguousArrayOfIntArray":
        # No type variables to map in this constraint
        return self


def verify_reshape_like_types(
    collapsed_type: ShapedType,
    expanded_type: ShapedType,
    reassociation: ArrayAttr[ArrayAttr[IntegerAttr]],
):
    """
    Verify that collapsed and expanded types conform to reassociation mapping.
    """
    expanded_rank = len(expanded_type.get_shape())
    collapsed_rank = len(collapsed_type.get_shape())

    if expanded_rank < collapsed_rank:
        raise VerifyException(
            f"expected the expanded type, {expanded_type} to have a higher (or same) rank "
            f"than the collapsed type, {collapsed_type}."
        )

    if collapsed_rank != len(reassociation):
        raise VerifyException(
            f"expected collapsed rank ({collapsed_rank}) to equal the number of "
            f"reassociation maps ({len(reassociation)})."
        )

    # Check that the total reassociation dimensions match the expanded type's rank.
    total_reassociation_dims = sum(len(rm) for rm in reassociation)
    if total_reassociation_dims != expanded_rank:
        raise VerifyException(
            f"expected the total number of reassociation dimensions ({total_reassociation_dims}) "
            f"to equal the expanded type's rank ({expanded_rank})."
        )

    verify_reshape_like_shapes_are_compatible(
        collapsed_shape=collapsed_type.get_shape(),
        expanded_shape=expanded_type.get_shape(),
        reassociation=reassociation,
    )


def verify_reshape_like_shapes_are_compatible(
    collapsed_shape: tuple[int, ...],
    expanded_shape: tuple[int, ...],
    reassociation: ArrayOfIntArrayAttr,
):
    """
    Verify that collapsed and expanded shapes adhere to reassociation mapping.
    """
    expanded_dim_start = 0

    for map_idx, rm in enumerate(reassociation):
        found_dynamic = False
        linearized_static = 1

        # Look at the next `len(rm)` dims in expanded_shape
        for dim in expanded_shape[expanded_dim_start : expanded_dim_start + len(rm)]:
            if dim == -1:
                found_dynamic = True
            else:
                linearized_static *= dim

        if found_dynamic:
            # if any is dynamic, the collapsed must be dynamic too
            if not collapsed_shape[map_idx] == -1:
                raise VerifyException(
                    f"expected dimension {map_idx} of collapsed type to be dynamic "
                    f"since one or more of the corresponding dimensions in the "
                    f"expanded type is dynamic"
                )
        else:
            # all static → product must match
            if collapsed_shape[map_idx] != linearized_static:
                raise VerifyException(
                    f"expected dimension {map_idx} of collapsed type to be static "
                    f"value of {linearized_static}"
                )

        expanded_dim_start += len(rm)
