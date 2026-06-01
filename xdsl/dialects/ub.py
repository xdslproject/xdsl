"""
The `ub` (undefined behavior) dialect.

Mirrors MLIR's
[UB dialect](https://mlir.llvm.org/docs/Dialects/UBOps/), which provides
operations and attributes for representing deferred undefined behavior.
"""

from __future__ import annotations

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import ConstantLike, Pure


@irdl_attr_definition
class PoisonAttr(ParametrizedAttribute):
    """
    Attribute carrying poison semantics for `ub.poison`.

    A default (empty) `#ub.poison` indicates a fully poisoned result. Other
    attributes (e.g. partially poisoned vectors) may be used to indicate
    additional poison semantics; in MLIR this is modelled via the
    `PoisonAttrInterface`.
    """

    name = "ub.poison"


@irdl_op_definition
class PoisonOp(IRDLOperation):
    """
    Materializes a compile-time poisoned constant value to indicate deferred
    undefined behavior.

    The `value` attribute indicates optional additional poison semantics (e.g.
    partially poisoned vectors); the default value indicates the result is
    fully poisoned.

    Examples:

    ```mlir
    // Short form (fully poisoned)
    %0 = ub.poison : i32
    // Long form (additional poison semantics)
    %1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
    ```
    """

    name = "ub.poison"

    value = prop_def(Attribute, default_value=PoisonAttr())

    result = result_def()

    traits = traits_def(ConstantLike(), Pure())

    assembly_format = "attr-dict (`<` $value^ `>`)? `:` type($result)"

    def __init__(
        self,
        result_type: Attribute,
        value: Attribute | None = None,
    ):
        super().__init__(
            result_types=[result_type],
            properties={"value": value} if value is not None else {},
        )


UB = Dialect(
    "ub",
    [
        PoisonOp,
    ],
    [
        PoisonAttr,
    ],
)
