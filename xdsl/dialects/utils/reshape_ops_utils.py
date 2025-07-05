"""
Utilities used by reshape ops.
See [MLIR counterpart](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h)
for more details.
"""

from dataclasses import dataclass

from typing_extensions import TypeVar

from xdsl.dialects.builtin import I64, Annotated, ArrayAttr, IntegerAttr
from xdsl.ir import Attribute
from xdsl.irdl import AtLeast, AttrConstraint, ConstraintContext, GenericAttrConstraint
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class ContiguousArrayOfIntArray(
    GenericAttrConstraint[ArrayAttr[ArrayAttr[IntegerAttr]]]
):
    """
    Enforce an ArrayAttr of ArrayAttr[IntegerAttr] to contain contiguous integer values across all inner arrays.
    For example: [[0, 1], [2, 3]] is valid, but [[3, 4], [0, 1]] is not.
    An empty inner array is considered contiguous.
    """

    def verify(
        self, attr: Attribute, constraint_context: ConstraintContext | None = None
    ) -> None:
        if not isa(
            attr,
            ArrayAttr[
                ArrayAttr[
                    Annotated[
                        IntegerAttr[I64],
                        IntegerAttr.constr(value=AtLeast(0)),
                    ]
                ]
            ],
        ):
            raise VerifyException(
                f"Expected ArrayAttr[ArrayAttr[IntegerAttr[I64]]] but got {getattr(attr, 'name', type(attr))}"
            )

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
