"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

import abc
from collections.abc import Iterable, Mapping, Sequence
from typing import Generic, Literal

from typing_extensions import TypeVar, cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    ContainerType,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    ShapedType,
    StaticShapeArrayConstr,
    StringAttr,
    TensorType,
    TupleType,
    TypedAttribute
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    opt_prop_def,
    param_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class EmitC_OpaqueType(ParametrizedAttribute, TypeAttribute):
    """EmitC opaque type"""

    name = "emitc.opaque"
    value: StringAttr

    def verify(self) -> None:
        if not self.value.data:
            raise VerifyException("expected non empty string in !emitc.opaque type")
        if self.value.data[-1] == "*":
            raise VerifyException(
                "pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead"
            )


@irdl_attr_definition
class EmitC_PtrDiffT(ParametrizedAttribute, TypeAttribute):
    """
    EmitC signed pointer diff type.
    Signed data type as wide as platform-specific pointer types. In particular, it is as wide as emitc.size_t.
    It corresponds to ptrdiff_t found in <stddef.h>.
    """

    name = "emitc.ptrdiff_t"


@irdl_attr_definition
class EmitC_SignedSizeT(ParametrizedAttribute, TypeAttribute):
    """
    EmitC signed size type.
    Data type representing all values of emitc.size_t, plus -1. It corresponds to ssize_t found in <sys/types.h>.
    Use of this type causes the code to be non-C99 compliant.
    """

    name = "emitc.ssize_t"


@irdl_attr_definition
class EmitC_SizeT(ParametrizedAttribute, TypeAttribute):
    """
    EmitC unsigned size type.
    Unsigned data type as wide as platform-specific pointer types. It corresponds to size_t found in <stddef.h>.
    """

    name = "emitc.size_t"


EmitCIntegerType = IntegerType[Literal[1, 8, 16, 32, 64]]
"""
Type for integer types supported by EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
"""

EmitCIntegerTypeConstr = irdl_to_attr_constraint(EmitCIntegerType)
"""
Constraint for integer types supported by EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
"""

EmitCFloatType = Float16Type | BFloat16Type | Float32Type | Float64Type
EmitCFloatTypeConstr = irdl_to_attr_constraint(EmitCFloatType)
"""
Supported floating-point type in EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L117)
"""

EmitCPointerWideType = EmitC_PtrDiffT | EmitC_SignedSizeT | EmitC_SizeT
EmitCPointerWideTypeConstr = irdl_to_attr_constraint(EmitCPointerWideType)
"""
Constraint for pointer-wide types supported by EmitC.
These types have the same width as platform-specific pointer types.
"""

EmitCIntegerIndexOpaqueType = EmitCIntegerType | IndexType | EmitC_OpaqueType
EmitCIntegerIndexOpaqueTypeConstr = irdl_to_attr_constraint(EmitCIntegerIndexOpaqueType)
"""
Constraint for integer, index, or opaque types supported by EmitC.
"""

EmitCArrayElementType = (
    EmitCIntegerIndexOpaqueType | EmitCFloatType | EmitCPointerWideType
)
EmitCArrayElementTypeConstr = irdl_to_attr_constraint(EmitCArrayElementType)
"""
Constraint for valid element types in EmitC arrays.
"""


@irdl_attr_definition
class EmitC_OpaqueAttr(ParametrizedAttribute):
    """
    An opaque attribute of which the value gets emitted as is.
    """

    name = "emitc.opaque"
    value: StringAttr


EmitCArrayElementTypeCovT = TypeVar(
    "EmitCArrayElementTypeCovT",
    bound=EmitCArrayElementType,
    covariant=True,
    default=EmitCArrayElementType,
)


@irdl_attr_definition
class EmitC_ArrayType(
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[EmitCArrayElementTypeCovT],
    Generic[EmitCArrayElementTypeCovT],
):
    """EmitC array type"""

    name = "emitc.array"

    shape: ArrayAttr[IntAttr] = param_def(StaticShapeArrayConstr)
    element_type: EmitCArrayElementTypeCovT

    def __init__(
        self,
        shape: Iterable[int | IntAttr],
        element_type: EmitCArrayElementType,
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__(shape, element_type)

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("EmitC array shape must not be empty")

        if isinstance(self.element_type, EmitC_ArrayType):
            raise VerifyException("nested EmitC arrays are not allowed")

        for dim_attr in self.shape.data:
            if dim_attr.data < 0:
                raise VerifyException(
                    "EmitC array dimensions must have non-negative size"
                )

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> EmitCArrayElementTypeCovT:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            shape, type = parser.parse_ranked_shape()
            return ArrayAttr(IntAttr(dim) for dim in shape), type

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(
                self.shape, lambda dim: printer.print_string(f"{dim.data}"), "x"
            )
            printer.print_string("x")
            printer.print_attribute(self.element_type)


class EmitCTypeConstraint(AttrConstraint):
    """
    Check if a type is supported by EmitC.
    See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L62).
    """

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if isa(attr, TensorType):
            # EmitC only supports tensors with static shapes
            if not attr.has_static_shape():
                raise VerifyException(f"Type {attr} is not a supported EmitC type")
            elem_type = attr.get_element_type()
            if isinstance(elem_type, EmitC_ArrayType):
                raise VerifyException("EmitC type cannot be a tensor of EmitC arrays")
            self.verify(elem_type, constraint_context)
            return

        if isa(attr, EmitC_ArrayType):
            elem_type = attr.get_element_type()
            self.verify(elem_type, constraint_context)
            return

        if isinstance(attr, EmitC_PointerType):
            self.verify(attr.pointee_type, constraint_context)
            return

        if isinstance(attr, TupleType):
            for t in attr.types:
                if isinstance(t, EmitC_ArrayType):
                    raise VerifyException(
                        "EmitC type cannot be a tuple of EmitC arrays"
                    )
                self.verify(t, constraint_context)
            return

        EmitCArrayElementTypeConstr.verify(attr, constraint_context)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint:
        # No type variables to map in this constraint
        return self


EmitCTypeConstr = EmitCTypeConstraint()


@irdl_attr_definition
class EmitC_LValueType(ParametrizedAttribute, TypeAttribute):
    """
    EmitC lvalue type.
    Values of this type can be assigned to and their address can be taken.
    See [tablegen definition](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L87)
    """

    name = "emitc.lvalue"
    value_type: Attribute = param_def(EmitCTypeConstr)

    def verify(self) -> None:
        """
        Verify the LValueType.
        See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L1095)
        """
        if isinstance(self.value_type, EmitC_ArrayType):
            raise VerifyException("!emitc.lvalue cannot wrap !emitc.array type")


@irdl_attr_definition
class EmitC_PointerType(ParametrizedAttribute, TypeAttribute):
    """EmitC pointer type"""

    name = "emitc.ptr"
    pointee_type: TypeAttribute

    def verify(self) -> None:
        if isinstance(self.pointee_type, EmitC_LValueType):
            raise VerifyException("pointers to lvalues are not allowed")


EmitCFundamentalType = IndexType | EmitCPointerWideType | EmitCIntegerType | EmitCFloatType | EmitC_PointerType

class EmitC_BinaryOperation(IRDLOperation, abc.ABC):
    """Base class for EmitC binary operations."""

    lhs = operand_def(EmitCTypeConstr)
    rhs = operand_def(EmitCTypeConstr)
    result = result_def(EmitCTypeConstr)

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
        )


@irdl_op_definition
class EmitC_AddOp(EmitC_BinaryOperation):
    """
    Addition operation.

    With the `emitc.add` operation the arithmetic operator + (addition) can
    be applied. Supports pointer arithmetic where one operand is a pointer
    and the other is an integer or opaque type.

    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.add %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 + v2;
    float* v6 = v3 + v4;
    ```
    """

    name = "emitc.add"

    def verify_(self) -> None:
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type

        if isa(lhs_type, EmitC_PointerType) and isa(rhs_type, EmitC_PointerType):
            raise VerifyException(
                "emitc.add requires that at most one operand is a pointer"
            )

        if (
            isa(lhs_type, EmitC_PointerType)
            and not isa(rhs_type, IntegerType | EmitC_OpaqueType)
        ) or (
            isa(rhs_type, EmitC_PointerType)
            and not isa(lhs_type, IntegerType | EmitC_OpaqueType)
        ):
            raise VerifyException(
                "emitc.add requires that one operand is an integer or of opaque "
                "type if the other is a pointer"
            )


@irdl_op_definition
class EmitC_ApplyOp(IRDLOperation):
    """Apply operation"""

    name = "emitc.apply"

    assembly_format = """
        $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
      """

    applicableOperator = prop_def(StringAttr)

    operand = operand_def(AnyAttr())

    result = result_def(EmitCTypeConstr)

    def verify_(self) -> None:
        applicable_operator = self.applicableOperator.data

        # Applicable operator must not be empty
        if not applicable_operator:
            raise VerifyException("applicable operator must not be empty")

        if applicable_operator not in ("&", "*"):
            raise VerifyException("applicable operator is illegal")

        operand_type = self.operand.type
        result_type = self.result.type

        if applicable_operator == "&":
            if not isinstance(operand_type, EmitC_LValueType):
                raise VerifyException(
                    "operand type must be an lvalue when applying `&`"
                )
            if not isinstance(result_type, EmitC_PointerType):
                raise VerifyException("result type must be a pointer when applying `&`")
        else:  # applicable_operator == "*"
            if not isinstance(operand_type, EmitC_PointerType):
                raise VerifyException(
                    "operand type must be a pointer when applying `*`"
                )

    def has_side_effects(self) -> bool:
        """Return True if the operation has side effects."""
        return self.applicableOperator.data == "*"


@irdl_op_definition
class EmitC_CallOpaqueOp(IRDLOperation):
    """
    The `emitc.call_opaque` operation represents a C++ function call. The callee can be an arbitrary non-empty string.
    The call allows specifying order of operands and attributes in the call as follows:

        - integer value of index type refers to an operand;
        - attribute which will get lowered to constant value in call;
    """

    name = "emitc.call_opaque"

    callee = prop_def(StringAttr)
    args = opt_prop_def(ArrayAttr)
    template_args = opt_prop_def(ArrayAttr)
    # The SSA‐value operands of the call
    call_args = var_operand_def()
    res = var_result_def()

    irdl_options = (ParsePropInAttrDict(),)
    assembly_format = (
        "$callee `(` $call_args `)` attr-dict `:` functional-type(operands, results)"
    )

    def __init__(
        self,
        callee: StringAttr | str,
        call_args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        args: ArrayAttr[Attribute] | None = None,
        template_args: ArrayAttr[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if isinstance(callee, str):
            callee = StringAttr(callee)
        super().__init__(
            properties={
                "callee": callee,
                "args": args,
                "template_args": template_args,
            },
            operands=[call_args],
            result_types=[result_types],
            attributes=attributes,
        )

    def verify_(self) -> None:
        if not self.callee.data:
            raise VerifyException("callee must not be empty")

        if self.args is not None:
            for arg in self.args.data:
                if isa(arg, IntegerAttr[IndexType]):
                    index = arg.value.data
                    if not (0 <= index < len(self.call_args)):
                        raise VerifyException("index argument is out of range")
                elif isinstance(arg, ArrayAttr):
                    # see https://github.com/llvm/llvm-project/blob/2eb733b5a6ab17a3ae812bb55c1c7c64569cadcd/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L342
                    # This part is referenced as a FIXME there.
                    raise VerifyException("array argument has no type")

        if self.template_args is not None:
            for t_arg in self.template_args.data:
                if not isa(
                    t_arg,
                    TypeAttribute | IntegerAttr | FloatAttr | EmitC_OpaqueAttr,
                ):
                    raise VerifyException("template argument has invalid type")

        for res_type in self.res.types:
            if isinstance(res_type, EmitC_ArrayType):
                raise VerifyException("cannot return array type")


# ===----------------------------------------------------------------------===
# ConstantOp
# ===----------------------------------------------------------------------===
EmitC_OpaqueOrTypedAttr = EmitC_OpaqueAttr | TypedAttribute


def isPointerWideType(type : TypeAttribute):
    return isa(type, EmitC_SignedSizeT | EmitC_SizeT | EmitC_PtrDiffT)


@irdl_attr_definition
class EmitCType(ParametrizedAttribute, TypeAttribute):
    """
    Type supported by EmitC
    """
    name = "emitc.base_t"


# Check that the type of the initial value is compatible with the operations
# result type.
def verifyInitializationAttribute(op: IRDLOperation, value: Attribute) -> None:
    assert len(op.results) == 1 and "operation must have 1 result"

    if isa(value, EmitC_OpaqueAttr):
        return

    if isa(value, StringAttr) or isinstance(value, StringAttr):
        raise VerifyException("string attributes are not supported, use #emitc.opaque instead")

    resultType = op.results[0].type
    if isinstance(resultType, EmitC_LValueType):
        resultType = resultType.value_type

    attrType = cast(TypedAttribute, value).get_type()

    if isPointerWideType(cast(TypeAttribute, resultType)):
        return

    if resultType != attrType:
        raise VerifyException("requires attribute to either be an #emitc.opaque attribute or it's type (" + str(attrType)+") to match the op's result type (" + str(resultType) + ")")

    return


@irdl_op_definition
class EmitC_ConstantOp(IRDLOperation):
    """
    The `emitc.constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `emitc.constant` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. Since folding is supported,
    it should not be used with pointers.
    """

    name = "emitc.constant"

    value = prop_def(EmitC_OpaqueOrTypedAttr)
    result = result_def(EmitCTypeConstr)

    assembly_format = " attr-dict $value `:` type(results)"

    irdl_options = (ParsePropInAttrDict(), )

    def __init__(
        self,
        value: EmitC_OpaqueOrTypedAttr | int
    ):
        res_type = ""
        if isinstance(value, int):
            value = IntegerAttr(value, IntegerType(32))
            res_type  = value.type
        else:
            res_type = EmitC_OpaqueType(StringAttr(str(value)))
        super().__init__(
            properties={
                "value": value
            },
            result_types=[res_type]
        )

    def verify_(self) -> None:
        value = self.value

        if isa(value, StringAttr):
            raise VerifyException("string attributes are not supported, use #emitc.opaque instead")

        verifyInitializationAttribute(self, value)

        if not value:
            raise VerifyException("value must not be empty")
        if isinstance(value, EmitC_OpaqueType):
            if not value.value.data:
                raise VerifyException("value must not be empty")


    def has_side_effects(self) -> bool:
        """Return True if the operation has side effects."""
        return isa(self.result.type, EmitCFundamentalType)

    '''
    @classmethod
    @abc.abstractmethod
    def fold(cls, op: Operation) -> Sequence[SSAValue | Attribute] | None:
        """
        Attempts to fold the operation. The fold method cannot modify the IR.
        Returns either an existing SSAValue or an Attribute for each result of the operation.
        When folding is unsuccessful, returns None.
        """
        raise NotImplementedError()
    '''


@irdl_op_definition
class EmitC_VariableOp(IRDLOperation):
    """
    The `emitc.variable` operation produces an SSA value equal to some value
    specified by an attribute. This can be used to form simple integer and
    floating point variables, as well as more exotic things like tensor
    variables. The `emitc.variable` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. If further supports the EmitC
    pointer type, whereas folding is not supported.
    The `emitc.variable` is emitted as a C/C++ local variable.
    """

    name = "emitc.variable"

    value = prop_def(EmitC_OpaqueOrTypedAttr)
    result = result_def(EmitC_ArrayType | EmitC_LValueType)

    assembly_format = " attr-dict $value `:` type(results)"

    def __init__(
        self,
        value: EmitC_OpaqueOrTypedAttr,
        result_types : Attribute
    ):
        super().__init__(
            properties={
                "value": value
            },
            result_types=[result_types]
        )

    def verify_(self) -> None:
        value = self.value
        if not value:
            raise VerifyException("'emitc.variable' op requires attribute 'value'")

        if value and not isa(value, EmitC_OpaqueOrTypedAttr):
            raise VerifyException("'emitc.variable' op attribute 'value' failed to satisfy constraint: An opaque attribute or TypedAttr instance")

    def has_side_effects(self) -> bool:
        return True


EmitC = Dialect(
    "emitc",
    [
        EmitC_AddOp,
        EmitC_ApplyOp,
        EmitC_CallOpaqueOp,
        EmitC_ConstantOp,
        EmitC_VariableOp
    ],
    [
        EmitC_ArrayType,
        EmitC_LValueType,
        EmitC_OpaqueAttr,
        EmitC_OpaqueType,
        EmitC_PointerType,
        EmitC_PtrDiffT,
        EmitC_SignedSizeT,
        EmitC_SizeT,
    ],
)
