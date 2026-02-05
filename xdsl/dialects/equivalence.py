"""
An embedding of equivalence classes in IR, for use in equality saturation with
non-destructive rewrites.

See the overview [notebook](https://xdsl.readthedocs.io/stable/marimo/equivalence.html).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import IntAttr
from xdsl.interfaces import ConstantLikeInterface
from xdsl.ir import Attribute, Dialect, OpResult, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    lazy_traits_def,
    opt_attr_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    ConstantLike,
    HasParent,
    IsTerminator,
    Pure,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import DiagnosticException, VerifyException

EQSAT_COST_LABEL = "eqsat_cost"
"""
Key used to store the cost of computing the result of an operation.
"""


@irdl_op_definition
class ConstantClassOp(IRDLOperation, ConstantLikeInterface):
    """An e-class representing a known constant value.
    For non-constant e-classes, use [ClassOp][xdsl.dialects.equivalence.ClassOp].
    """

    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "equivalence.const_class"

    assembly_format = (
        "$arguments ` ` `(` `constant` `=` $value `)` attr-dict `:` type($result)"
    )
    traits = traits_def(Pure())

    arguments = var_operand_def(T)
    result = result_def(T)
    value = prop_def()
    min_cost_index = opt_attr_def(IntAttr)

    def get_constant_value(self):
        return self.value

    def __init__(self, const_arg: OpResult):
        if (trait := const_arg.owner.get_trait(ConstantLike)) is None:
            raise DiagnosticException(
                "The argument of a ConstantClass must be a constant-like operation."
            )
        value = trait.get_constant_value(const_arg.owner)
        super().__init__(
            operands=[const_arg],
            result_types=[const_arg.type],
            properties={"value": value},
        )


@irdl_op_definition
class ClassOp(IRDLOperation):
    """An e-class representing a set of equivalent values.
    E-classes that represent a constant value can instead
    be represented by [ConstantClassOp][xdsl.dialects.equivalence.ConstantClassOp].
    """

    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "equivalence.class"
    arguments = var_operand_def(T)
    result = result_def(T)
    min_cost_index = opt_attr_def(IntAttr)
    traits = traits_def(Pure())

    assembly_format = "$arguments attr-dict `:` type($result)"

    def __init__(
        self,
        *arguments: SSAValue,
        min_cost_index: IntAttr | None = None,
        res_type: Attribute | None = None,
    ):
        if not arguments:
            raise DiagnosticException("E-class op must have at least one operand")
        if res_type is None:
            res_type = arguments[0].type

        super().__init__(
            operands=[arguments],
            result_types=[res_type],
            attributes={"min_cost_index": min_cost_index},
        )

    def verify_(self) -> None:
        if not self.operands:
            raise VerifyException("E-class operations must have at least one operand.")

        for operand in self.operands:
            if isinstance(operand.owner, ClassOp):
                # The two ops should have been merged into one.
                raise VerifyException(
                    "A result of an e-class operation cannot be used as an operand of "
                    "another e-class."
                )

            if not operand.has_one_use():
                if len(set(use.operation for use in operand.uses)) == 1:
                    raise VerifyException(
                        "E-class operands must only be used once by the e-class."
                    )
                else:
                    raise VerifyException(
                        "E-class operands must only be used by the e-class."
                    )


AnyClassOp = ClassOp | ConstantClassOp
"""
A type representing either a [regular e-class operation][xdsl.dialects.equivalence.ClassOp]
or a [constant e-class operation][xdsl.dialects.equivalence.ConstantClassOp].
"""


@irdl_op_definition
class GraphOp(IRDLOperation):
    name = "equivalence.graph"

    inputs = var_operand_def()
    outputs = var_result_def()
    body = region_def()

    traits = lazy_traits_def(lambda: (SingleBlockImplicitTerminator(YieldOp),))

    assembly_format = (
        "($inputs^ `:` type($inputs))? `->` type($outputs) $body attr-dict"
    )

    def __init__(
        self,
        result_types: Sequence[Attribute] | None,
        body: Region,
    ):
        super().__init__(
            result_types=(result_types,),
            regions=[body],
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "equivalence.yield"
    values = var_operand_def()

    traits = traits_def(HasParent(GraphOp), IsTerminator())

    assembly_format = "$values `:` type($values) attr-dict"

    def __init__(
        self,
        *values: SSAValue,
    ):
        super().__init__(operands=[values])


Equivalence = Dialect(
    "equivalence",
    [
        ClassOp,
        ConstantClassOp,
        YieldOp,
        GraphOp,
    ],
)
