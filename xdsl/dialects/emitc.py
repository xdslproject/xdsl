"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

import abc
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Generic, Literal, Optional

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
    TypedAttribute,
    UnitAttr
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute
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
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    IsolatedFromAbove,
    NoTerminator,
    MemoryAllocEffect,
    Pure,
    RecursiveMemoryEffect,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

'''
[AutomaticAllocationScope, IsolatedFromAbove,
                         OpAsmOpInterface, SymbolTable,
                         Symbol]#GraphRegionNoTerminator.traits'''

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


class EmitC_UnaryOperation(IRDLOperation, abc.ABC):
    """Base class for EmitC unary operations."""

    operand = operand_def(EmitCTypeConstr)
    result = result_def(EmitCTypeConstr)

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    def __init__(
        self,
        operand: SSAValue,
        result_type: Attribute
    ):
        super().__init__(
            operands=[operand],
            result_types=[result_type]
        )

    def has_side_effects(self) -> bool:
        """If operand is fundamental type, the operation is pure."""
        return not isa(self.operand.type, EmitCFundamentalType)


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
class EmitC_AddressOfOp(IRDLOperation):
    name = "emitc.address_of"
    operand = operand_def(EmitC_LValueType)
    result = result_def(EmitCTypeConstr)

    def verify_(self) -> None:
        if not isinstance(self.operand.type, EmitC_LValueType):
            raise VerifyException(
                "operand type must be an lvalue when applying `&`"
            )
        if not isinstance(self.result.type, EmitC_PointerType):
            raise VerifyException("result type must be a pointer when applying `&`")

    def has_side_effects(self) -> bool:
        """Return True if the operation has side effects."""
        return False


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
class EmitC_AssignOp(IRDLOperation):
    """
    Assign operation.

    The `emitc.assign` operation stores an SSA value to the location designated by an
    EmitC variable. This operation doesn't return any value. The assigned value
    must be of the same type as the variable being assigned. The operation is
    emitted as a C/C++ '=' operator.
    """

    name = "emitc.assign"

    var = operand_def(EmitC_LValueType)
    value = operand_def(EmitCTypeConstr)

    assembly_format = "$value `:` type($value) `to` $var `:` type($var) attr-dict"

    def __init__(
        self,
        var: SSAValue,
        value: SSAValue,
    ):
        super().__init__(
            operands=[var, value],
            result_types=[]
        )

    def verify_(self) -> None:
        if self.var.type != EmitC_LValueType(self.value.type):
            raise VerifyException("'emitc.assign' op operands var and value must have the same type")


@irdl_op_definition
class EmitC_BitwiseAndOp(EmitC_BinaryOperation):
    """
    Bitwise and operation.

    With the `emitc.bitwise_and` operation the bitwise operator & (and) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_and %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 & v2;
    ```
    """

    name = "emitc.bitwise_and"


@irdl_op_definition
class EmitC_BitwiseLeftShiftOp(EmitC_BinaryOperation):
    """
    Bitwise left shift operation.

    With the `emitc.bitwise_left_shift` operation the bitwise operator <<
    (left shift) can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 << v2;
    ```
    """

    name = "emitc.bitwise_left_shift"


@irdl_op_definition
class EmitC_BitwiseNotOp(EmitC_UnaryOperation):
    """
    Bitwise not operation.

    With the `emitc.bitwise_not` operation the bitwise operator ~ (not) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_not %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = ~v1;
    ```
    """

    name = "emitc.bitwise_not"


@irdl_op_definition
class EmitC_BitwiseOrOp(EmitC_BinaryOperation):
    """
    Bitwise or operation.

    With the `emitc.bitwise_or` operation the bitwise operator | (or)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_or %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 | v2;
    ```
    """

    name = "emitc.bitwise_or"


@irdl_op_definition
class EmitC_BitwiseRightShiftOp(EmitC_BinaryOperation):
    """
    Bitwise right shift operation.

    With the `emitc.bitwise_right_shift` operation the bitwise operator >>
    (right shift) can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 >> v2;
    ```
    """

    name = "emitc.bitwise_right_shift"


@irdl_op_definition
class EmitC_BitwiseXorOp(EmitC_BinaryOperation):
    """
    Bitwise xor operation.

    With the `emitc.bitwise_xor` operation the bitwise operator ^ (xor)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 ^ v2;
    ```
    """

    name = "emitc.bitwise_xor"


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


@irdl_op_definition
class EmitC_CastOp(IRDLOperation):
    """
    Cast operation.

    The `emitc.cast` operation performs an explicit type conversion and is emitted
    as a C-style cast expression. It can be applied to integer, float, index
    and EmitC types.

    Example:

    ```mlir
    // Cast from `int32_t` to `float`
    %0 = emitc.cast %arg0: i32 to f32

    // Cast from `void` to `int32_t` pointer
    %1 = emitc.cast %arg1 :
        !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
    ```
    """

    name = "emitc.cast"

    assembly_format = "$source attr-dict `:` type($source) `to` type($dest)"

    source = operand_def(EmitCTypeConstr)
    dest = result_def(EmitCTypeConstr)

    def has_side_effects(self) -> bool:
        return False


@irdl_op_definition
class EmitC_ClassOp(IRDLOperation):
    """
    Represents a C++ class definition, encapsulating fields and methods.

    The `emitc.class` operation defines a C++ class, acting as a container
    for its data fields (`emitc.field`) and methods (`emitc.func`).
    It creates a distinct scope, isolating its contents from the surrounding
    MLIR region, similar to how C++ classes encapsulate their internals.

    Example:

    ```mlir
    emitc.class @modelClass {
      emitc.field @fieldName0 : !emitc.array<1xf32> = {emitc.opaque = "input_tensor"}
      emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @fieldName0 : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
      }
    }
    // Class with a final specifer
    emitc.class final @modelClass {
      emitc.field @fieldName0 : !emitc.array<1xf32> = {emitc.opaque = "input_tensor"}
      emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @fieldName0 : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
      }
    }
    ```
    """

    name = "emitc.class"

    assembly_format = "(`final` $final_specifier^)? $sym_name attr-dict-with-keyword $body"

    sym_name = prop_def(StringAttr)
    final_specifier = prop_def(UnitAttr)

    body = region_def()

    traits = traits_def(
        SymbolTable(),
        IsolatedFromAbove(),
        NoTerminator(),
        MemoryAllocEffect()
    )

    def get_block(self):
        if self.body.block:
            return self.body.block


@irdl_op_definition
class EmitC_FieldOp(IRDLOperation):
    """
    A field within a class.

    The `emitc.field` operation declares a named field within an `emitc.class`
    operation. The field's type must be an EmitC type.

    Example:

    ```mlir
    // Example with an attribute:
    emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.opaque = "another_feature"}
    // Example with no attribute:
    emitc.field @fieldName0 : !emitc.array<1xf32>
    // Example with an initial value:
    emitc.field @fieldName0 : !emitc.array<1xf32> = dense<0.0>
    // Example with an initial value and attributes:
    emitc.field @fieldName0 : !emitc.array<1xf32> = dense<0.0> {
      emitc.opaque = "input_tensor"}
    """

    name = "emitc.field"

    sym_name = prop_def(StringAttr)
    type = prop_def(TypeAttribute)
    initial_value = opt_prop_def(EmitC_OpaqueAttr | TypedAttribute)

    assembly_format = "$sym_name `:` custom<EmitCFieldOpTypeAndInitialValue>($type, $initial_value) attr-dict"

    def verify_(self) -> None:
        parentOp = self.parent_op()
        if not parentOp or not isinstance(parentOp, EmitC_ClassOp):
            raise VerifyException("field must be nested within an emitc.class operation")
        name = self.sym_name
        if not name or name.data == "":
            raise VerifyException("field must have a non-empty symbol name")


@irdl_op_definition
class EmitC_GetFieldOp(IRDLOperation):
    """
    Obtain access to a field within a class instance.

    The `emitc.get_field` operation retrieves the lvalue of a
     named field from a given class instance.

     Example:

     ```mlir
     %0 = get_field @fieldName0 : !emitc.array<1xf32>
     ```
    """

    name = "emitc.get_field"

    field_name = prop_def(StringAttr)
    result = result_def(EmitCTypeConstr)
    assembly_format = "$field_name `:` type($result) attr-dict"

    def verify_(self) -> None:
        parentOp = self.parent_op()
        if not parentOp or not isinstance(parentOp, EmitC_ClassOp):
            raise VerifyException(" must be nested within an emitc.class operation")


class CmpPredicate(StrEnum):
    eq = "eq"
    ne = "ne"
    lt = "lt"
    le = "le"
    gt = 'gt'
    ge = "ge"
    three_way = "three_way"

@irdl_attr_definition
class EmitC_CmpPredicateAttr(
        EnumAttribute[CmpPredicate] # pyright: ignore[reportInvalidTypeArguments]
    ):
    name = "emitc.cmp_predicate"


@irdl_op_definition
class EmitC_CmpOp(EmitC_BinaryOperation):
    """
    Comparison operation.

    With the `emitc.cmp` operation the comparison operators ==, !=, <, <=, >, >=, <=>
    can be applied.

    Its first argument is an attribute that defines the comparison operator:

    - equal to (mnemonic: `"eq"`; integer value: `0`)
    - not equal to (mnemonic: `"ne"`; integer value: `1`)
    - less than (mnemonic: `"lt"`; integer value: `2`)
    - less than or equal to (mnemonic: `"le"`; integer value: `3`)
    - greater than (mnemonic: `"gt"`; integer value: `4`)
    - greater than or equal to (mnemonic: `"ge"`; integer value: `5`)
    - three-way-comparison (mnemonic: `"three_way"`; integer value: `6`)

    Example:
    ```mlir
    // Custom form of the cmp operation.
    %0 = emitc.cmp eq, %arg0, %arg1 : (i32, i32) -> i1
    %1 = emitc.cmp lt, %arg2, %arg3 :
        (
          !emitc.opaque<"std::valarray<float>">,
          !emitc.opaque<"std::valarray<float>">
        ) -> !emitc.opaque<"std::valarray<bool>">
    ```
    ```c++
    // Code emitted for the operations above.
    bool v5 = v1 == v2;
    std::valarray<bool> v6 = v3 < v4;
    ```
    """

    name = "emitc.cmp"

    predicate = prop_def(EmitC_CmpPredicateAttr)
    lhs = operand_def(EmitCTypeConstr)
    rhs = operand_def(EmitCTypeConstr)
    result = result_def(EmitCTypeConstr)

    assembly_format = "$predicate `,` operands attr-dict `:` functional-type(operands, results)"

    def __init__(
        self,
        pred : int,
        lhs : SSAValue,
        rhs : SSAValue,
        result_type : Attribute
    ):
        super().__init__(
            lhs, rhs, result_type,
        )
        self.predicate = EmitC_CmpPredicateAttr(CmpPredicate.gt)


@irdl_op_definition
class EmitC_ConditionalOp(IRDLOperation):
    """
    Conditional (ternary) operation.

    With the `emitc.conditional` operation the ternary conditional operator can
    be applied.

    Example:

    ```mlir
    %0 = emitc.cmp gt, %arg0, %arg1 : (i32, i32) -> i1

    %c0 = "emitc.constant"() {value = 10 : i32} : () -> i32
    %c1 = "emitc.constant"() {value = 11 : i32} : () -> i32

    %1 = emitc.conditional %0, %c0, %c1 : i32
    ```
    ```c++
    // Code emitted for the operations above.
    bool v3 = v1 > v2;
    int32_t v4 = 10;
    int32_t v5 = 11;
    int32_t v6 = v3 ? v4 : v5;
    ```
    """

    name = "emitc.conditional"

    condition = operand_def(IntegerType(1))
    true_value = operand_def(EmitCTypeConstr)
    false_value = operand_def(EmitCTypeConstr)
    result = result_def(EmitCTypeConstr)

    #assembly_format = "operands attr-dict `:` type($result)"

    def has_side_effects(self) -> bool:
        return False


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

    def __init__(
        self,
        value: EmitC_OpaqueOrTypedAttr | IntegerAttr
    ):
        if isinstance(value, IntegerAttr):
            res_type = value.type # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        else:
            res_type = EmitC_OpaqueType(StringAttr("std::string"))
        super().__init__(
            properties={ "value": value }, # pyright: ignore[reportUnknownArgumentType]
            result_types=[res_type] # pyright: ignore[reportUnknownArgumentType]
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
class EmitC_DereferenceOp(IRDLOperation):
    name = "emitc.dereference"
    operand = operand_def(EmitC_PointerType)
    result = result_def(EmitC_LValueType)

    def verify_(self) -> None:
        if not isinstance(self.operand.type, EmitC_PointerType):
            raise VerifyException(
                "operand type must be a pointer when applying `*`"
            )

    def has_side_effects(self) -> bool:
        """Return True if the operation has side effects."""
        return True


@irdl_op_definition
class EmitC_DivOp(EmitC_BinaryOperation):
    """
    Division operation.

    With the `emitc.div` operation the arithmetic operator / (division) can
    be applied.

    Example:

    ```mlir
    // Custom form of the division operation.
    %0 = emitc.div %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.div %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 / v2;
    float v6 = v3 / v4;
    ```
    """

    name = "emitc.div"

    lhs = operand_def(EmitCFloatType | EmitCIntegerType | IndexType | EmitC_OpaqueType)
    rhs = operand_def(EmitCFloatType | EmitCIntegerType | IndexType | EmitC_OpaqueType)
    result = result_def(EmitCFloatType | EmitCIntegerType | IndexType | EmitC_OpaqueType)

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result_type: Attribute
    ):
        super().__init__(
            lhs,
            rhs,
            result_type
        )

class EmitC_YieldOp(IRDLOperation):
    """
    The `emitc.yield` terminates its parent EmitC op's region, optionally yielding
    an SSA value. The semantics of how the values are yielded is defined by the
    parent operation.
    If `emitc.yield` has an operand, the operand must match the parent operation's
    result. If the parent operation defines no values, then the `emitc.yield`
    may be left out in the custom syntax and the builders will insert one
    implicitly. Otherwise, it has to be present in the syntax to indicate which
    value is yielded.
    """

    name = "emitc.yield"

    result = result_def(EmitCTypeConstr)

    assembly_format = "[{ attr-dict ($result^ `:` type($result))? }]"

    def verify_(self) -> None:
        pass


@irdl_op_definition
class EmitC_IfOp(IRDLOperation):
    """
    If-then-else operation.

    The `emitc.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    emitc.if %b  {
      ...
    } else {
      ...
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    blocks. The blocks are always terminated with `emitc.yield`, which can be
    left out to be inserted implicitly. This operation doesn't produce any
    results.
    """

    name = "emitc.if"
    condition = operand_def(IntegerType(1))
    thenRegion = region_def("single_block")
    elseRegion = region_def()

    traits = traits_def(
        RecursiveMemoryEffect(), NoTerminator()
    )

    def __init__(
        self,
        cond: SSAValue | Operation,
        thenRegion: Region | Sequence[Block] | Sequence[Operation],
        elseRegion: Region | Sequence[Block] | Sequence[Operation] | None = None,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        if elseRegion is None:
            elseRegion = Region()

        super().__init__(
            operands=[cond],
            result_types=[],
            regions=[thenRegion, elseRegion],
            attributes=attr_dict
        )

    irdl_options = (ParsePropInAttrDict(), )
    assembly_format = "attr-dict $condition  $thenRegion (`else` $elseRegion^)?"


@irdl_op_definition
class EmitC_IncludeOp(IRDLOperation):
    """
    Include operation.

    The `emitc.include` operation allows to define a source file inclusion via the
    `#include` directive.

    Example:

    ```mlir
    // Custom form defining the inclusion of `<myheader>`.
    emitc.include <"myheader.h">

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h", is_standard_include} : () -> ()

    // Custom form defining the inclusion of `"myheader"`.
    emitc.include "myheader.h"

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h"} : () -> ()
    ```
    """

    name = "emitc.include"
    include = prop_def(StringAttr)
    is_standard_include = opt_prop_def(UnitAttr)

    irdl_options = (ParsePropInAttrDict(), )

    assembly_format = "attr-dict ` `(`<` $is_standard_include^)? $include `>`"

    def __init__(
        self,
        include : StringAttr,
        is_standard_include : Optional[UnitAttr] = None
    ):
        super().__init__(
            operands=[],
            properties={
                "include": include,
                "is_standard_include" : is_standard_include
            }
        )


@irdl_op_definition
class EmitC_LiteralOp(IRDLOperation):
    """
    Literal operation.

    The `emitc.literal` operation produces an SSA value equal to some constant
    specified by an attribute.

    Example:

    ```mlir
    %p0 = emitc.literal "M_PI" : f32
    %1 = "emitc.add" (%arg0, %p0) : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operation above.
    float v2 = v1 + M_PI;
    ```
    """

    name = "emitc.literal"

    value = prop_def(StringAttr)
    result = result_def(EmitCTypeConstr)

    traits = traits_def(Pure())

    assembly_format = "$value attr-dict `:` type($result)"

    def verify_(self):
        if self.value.data == "":
            raise VerifyException("value must not be empty")


@irdl_op_definition
class EmitC_LoadOp(IRDLOperation):
    """
    Load an lvalue into an SSA value.

    This operation loads the content of a modifiable lvalue into an SSA value.
    Modifications of the lvalue executed after the load are not observable on
    the produced value.

    Example:

    ```mlir
    %1 = emitc.load %0 : !emitc.lvalue<i32>
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = v1;
    ```
    """

    name = "emitc.load"

    operand = operand_def(EmitC_LValueType)
    result = result_def(EmitCTypeConstr)

    irdl_options = (ParsePropInAttrDict(),)

    #assembly_format = "$operand attr-dict `:` type($operand)"

    def __init__(self,
        op: SSAValue
        ):
        op_type = op.type
        assert(isinstance(op_type, EmitC_LValueType))
        res_type = op_type.value_type
        super().__init__(
            operands=[op],
            result_types=[res_type]
            )


@irdl_op_definition
class EmitC_LogicalAndOp(EmitC_BinaryOperation):
    """
    Logical and operation.

    With the `emitc.logical_and` operation the logical operator && (and) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.logical_and %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 && v2;
    ```
    """

    name = "emitc.logical_and"

    result = result_def(IntegerType(1))

    assembly_format = "operands attr-dict `:` type(operands)"


@irdl_op_definition
class EmitC_LogicalNotOp(EmitC_UnaryOperation):
    """
    Logical not operation.

    With the `emitc.logical_not` operation the logical operator ! (negation) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.logical_not %arg0 : i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v2 = !v1;
    ```
    """

    name = "emitc.logical_not"

    result = result_def(IntegerType(1))

    assembly_format = "operands attr-dict `:` type(operands)"


@irdl_op_definition
class EmitC_LogicalOrOp(EmitC_BinaryOperation):
    """
    Logical or operation.

    With the `emitc.logical_or` operation the logical operator || (inclusive or)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.logical_or %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 || v2;
    ```
    """

    name = "emitc.logical_or"

    result = result_def(IntegerType(1))

    assembly_format = "operands attr-dict `:` type(operands)"


@irdl_op_definition
class EmitC_MemberOfPtrOp(IRDLOperation):
    """
    Member of pointer operation.

    With the `emitc.member_of_ptr` operation the member access operator `->`
    can be applied.

    Example:

    ```mlir
    %0 = "emitc.member_of_ptr" (%arg0) {member = "a"}
        : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>)
        -> !emitc.lvalue<i32>
    %1 = "emitc.member_of_ptr" (%arg0) {member = "b"}
        : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>)
        -> !emitc.array<2xi32>
    ```
    """

    name = "emitc.member_of_ptr"

    member = prop_def(StringAttr)
    operand = operand_def(EmitC_LValueType)
    result = result_def(EmitC_ArrayType | EmitC_LValueType)

    def __init__(
          self,
          operand: SSAValue,
          member: str,
          result_type: Attribute
    ):
        if not isinstance(operand.type, EmitC_OpaqueAttr | EmitC_PointerType):
            raise VerifyException("emitc.member_of_ptr expects an opaque or pointer operand type")

        super().__init__(
            operands=[operand],
            properties={"member" : StringAttr(member)},
            result_types=[result_type]
        )


@irdl_op_definition
class EmitC_MemberOp(IRDLOperation):
    """
    Member operation.

    With the `emitc.member` operation the member access operator `.` can be
    applied.

    Example:

    ```mlir
    %0 = "emitc.member" (%arg0) {member = "a"}
        : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
    %1 = "emitc.member" (%arg0) {member = "b"}
        : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.array<2xi32>
    ```
    """

    name = "emitc.member"

    member = prop_def(StringAttr)
    operand = operand_def(EmitC_LValueType)
    result = result_def(EmitC_ArrayType | EmitC_LValueType)

    def __init__(
          self,
          operand: SSAValue,
          member: str,
          result_type: Attribute
    ):
        if not isinstance(operand.type, EmitC_OpaqueAttr):
            raise VerifyException("emitc.member expects an opaque operand type")

        super().__init__(
            operands=[operand],
            properties={"member" : StringAttr(member)},
            result_types=[result_type]
        )


@irdl_op_definition
class EmitC_MulOp(EmitC_BinaryOperation):
    """
    Multiplication operation.

    With the `emitc.mul` operation the arithmetic operator * (multiplication) can
    be applied.

    Example:

    ```mlir
    // Custom form of the multiplication operation.
    %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.mul %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 * v2;
    float v6 = v3 * v4;
    ```
    """

    name = "emitc.mul"

    lhs = operand_def(EmitCFloatType | EmitCIntegerType | IndexType | EmitC_OpaqueType)
    rhs = operand_def(EmitCFloatType | EmitCIntegerType | IndexType | EmitC_OpaqueType)
    result = result_def(EmitCFloatType | EmitCIntegerType | IndexType | EmitC_OpaqueType)

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result_type: Attribute
    ):
        super().__init__(
            lhs,
            rhs,
            result_type
        )


@irdl_op_definition
class EmitC_RemOp(EmitC_BinaryOperation):
    """
    Remainder operation.

    With the `emitc.rem` operation the arithmetic operator % (remainder) can
    be applied.

    Example:

    ```mlir
    // Custom form of the remainder operation.
    %0 = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v5 = v1 % v2;
    ```
    """

    name = "emitc.rem"

    lhs = operand_def(EmitCIntegerType | IndexType | EmitC_OpaqueType)
    rhs = operand_def(EmitCIntegerType | IndexType | EmitC_OpaqueType)
    result = result_def(EmitCIntegerType | IndexType | EmitC_OpaqueType)

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result_type: Attribute
    ):
        super().__init__(
            lhs,
            rhs,
            result_type
        )


@irdl_op_definition
class EmitC_SubOp(EmitC_BinaryOperation):
    """
    Subtraction operation.

    With the `emitc.sub` operation the arithmetic operator - (subtraction) can
    be applied.

    Example:

    ```mlir
    // Custom form of the subtraction operation.
    %0 = emitc.sub %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.sub %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    %2 = emitc.sub %arg4, %arg5 : (!emitc.ptr<i32>, !emitc.ptr<i32>)
        -> !emitc.ptrdiff_t
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v7 = v1 - v2;
    float* v8 = v3 - v4;
    ptrdiff_t v9 = v5 - v6;
    ```
    """

    name = "emitc.sub"

    def verify_(self) -> None:
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type

        if isa(lhs_type, EmitC_PointerType) and isa(rhs_type, EmitC_PointerType):
            raise VerifyException(
                "emitc.sub requires that at most one operand is a pointer"
            )

        if (
            isa(lhs_type, EmitC_PointerType)
            and not isa(rhs_type, IntegerType | EmitC_OpaqueType)
        ) or (
            isa(rhs_type, EmitC_PointerType)
            and not isa(lhs_type, IntegerType | EmitC_OpaqueType)
        ):
            raise VerifyException(
                "emitc.sub requires that one operand is an integer or of opaque "
                "type if the other is a pointer"
            )


@irdl_op_definition
class EmitC_SubscriptOp(IRDLOperation):
    """
    Subscript operation.

    With the `emitc.subscript` operation the subscript operator `[]` can be applied
    to variables or arguments of array, pointer and opaque type.

    Example:

    ```mlir
    %i = index.constant 1
    %j = index.constant 7
    %0 = emitc.subscript %arg0[%i, %j] : (!emitc.array<4x8xf32>, index, index)
           -> !emitc.lvalue<f32>
    %1 = emitc.subscript %arg1[%i] : (!emitc.ptr<i32>, index)
           -> !emitc.lvalue<i32>
    ```
    """

    name = "emitc.subscript"

    value = operand_def(EmitC_ArrayType | EmitC_OpaqueType | EmitC_PointerType)
    indices = var_operand_def(EmitCTypeConstr)
    result = result_def(EmitC_LValueType)

    assembly_format = "$value `[` $indices `]` attr-dict `:` functional-type(operands, results)"

    def __init__(
        self,
        value: SSAValue,
        indices: Sequence[SSAValue]
    ):
        # array[i]
        if isinstance(value.type, EmitC_ArrayType):
            res_type = EmitC_LValueType(value.type.element_type) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        # ptr[i]
        elif isinstance(value.type, EmitC_PointerType):
            res_type = EmitC_LValueType(value.type.pointee_type)

        # opaque
        elif isinstance(value.type, EmitC_OpaqueType):
            res_type = value.type

        else:
            raise VerifyException(f"Unsupported type for emitc.subscript: {value.type}")

        super().__init__(
            operands=[value, *indices],
            result_types=[res_type]
        )

    def verify_(self) -> None:
        val_type = self.value.type

        # array[i]
        if isinstance(val_type, EmitC_ArrayType):
            if len(self.indices) != len(val_type.shape):
                raise VerifyException(f"Array subscript expects {len(val_type.shape)} indices, got {len(self.indices)}")

        # ptr[i]
        if isinstance(val_type, EmitC_PointerType):
            if len(self.indices) != 1:
                raise VerifyException(f"Pointer subscript expects exactly one index")

        # opaque
        if isinstance(val_type, EmitC_OpaqueType):
            if len(self.indices) == 0:
                raise VerifyException("Opaque subscript expects at least one index")


@irdl_op_definition
class EmitC_UnaryMinusOp(EmitC_UnaryOperation):
    """
    Unary minus operation.

    With the `emitc.unary_minus` operation the unary operator - (minus) can be
    applied.

    Example:

    ```mlir
    %0 = emitc.unary_minus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = -v1;
    ```
    """

    name = "emitc.unary_minus"


@irdl_op_definition
class EmitC_UnaryPlusOp(EmitC_UnaryOperation):
    """
    Unary plus operation.

    With the `emitc.unary_plus` operation the unary operator + (plus) can be
    applied.

    Example:

    ```mlir
    %0 = emitc.unary_plus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = +v1;
    ```
    """

    name = "emitc.unary_plus"


@irdl_op_definition
class EmitC_VariableOp(IRDLOperation):
    """
    Variable operation.

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

    #assembly_format = " attr-dict $value `:` type(results)"

    def __init__(
        self,
        value: EmitC_OpaqueOrTypedAttr | IntegerAttr,
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


@irdl_op_definition
class EmitC_VerbatimOp(IRDLOperation):
    """
    Verbatim operation.

    The `emitc.verbatim` operation produces no results and the value is emitted as is
    followed by a line break  ('\n' character) during translation.

    Note: Use with caution. This operation can have arbitrary effects on the
    semantics of the emitted code. Use semantically more meaningful operations
    whenever possible. Additionally this op is *NOT* intended to be used to
    inject large snippets of code.

    This operation can be used in situations where a more suitable operation is
    not yet implemented in the dialect or where preprocessor directives
    interfere with the structure of the code. One example of this is to declare
    the linkage of external symbols to make the generated code usable in both C
    and C++ contexts:

    ```c++
    #ifdef __cplusplus
    extern "C" {
    #endif

    ...

    #ifdef __cplusplus
    }
    #endif
    ```

    If the `emitc.verbatim` op has operands, then the `value` is interpreted as
    format string, where `{}` is a placeholder for an operand in their order.
    For example, `emitc.verbatim "#pragma my src={} dst={}" %src, %dest : i32, i32`
    would be emitted as `#pragma my src=a dst=b` if `%src` became `a` and
    `%dest` became `b` in the C code.
    `{{` in the format string is interpreted as a single `{` and doesn't introduce
    a placeholder.

    Example:

    ```mlir
    emitc.verbatim "typedef float f32;"
    emitc.verbatim "#pragma my var={} property" args %arg : f32
    ```
    ```c++
    // Code emitted for the operation above.
    typedef float f32;
    #pragma my var=v1 property
    ```
    """

    name = "emitc.verbatim"

    value = prop_def(StringAttr)
    fmtArgs = var_operand_def(AnyAttr())

    assembly_format = "$value (`args` $fmtArgs^ `:` type($fmtArgs))? attr-dict"

    def __init__(self,
        value: StringAttr,
        operands: Sequence[AnyAttr],
    ):
        super().__init__(
            operands=[operands], # pyright: ignore[reportArgumentType]
            properties={
                "value" : value
            }
        )

    def verify_(self) -> None:
        value = self.value
        if not value:
            raise VerifyException("'emitc.verbatim' op requires attribute 'value'")


EmitC = Dialect(
    "emitc",
    [
        EmitC_AddOp,
        EmitC_AddressOfOp,
        EmitC_ApplyOp,
        EmitC_AssignOp,
        EmitC_BitwiseAndOp,
        EmitC_BitwiseLeftShiftOp,
        EmitC_BitwiseNotOp,
        EmitC_BitwiseOrOp,
        EmitC_BitwiseRightShiftOp,
        EmitC_BitwiseXorOp,
        EmitC_CallOpaqueOp,
        EmitC_CastOp,
        EmitC_ClassOp,
        EmitC_FieldOp,
        EmitC_GetFieldOp,
        EmitC_CmpOp,
        EmitC_ConditionalOp,
        EmitC_ConstantOp,
        EmitC_DereferenceOp,
        EmitC_DivOp,
        EmitC_IfOp,
        EmitC_IncludeOp,
        EmitC_LiteralOp,
        EmitC_LoadOp,
        EmitC_LogicalAndOp,
        EmitC_LogicalNotOp,
        EmitC_LogicalOrOp,
        EmitC_MemberOfPtrOp,
        EmitC_MemberOp,
        EmitC_MulOp,
        EmitC_RemOp,
        EmitC_SubOp,
        EmitC_SubscriptOp,
        EmitC_UnaryMinusOp,
        EmitC_UnaryPlusOp,
        EmitC_VariableOp,
        EmitC_VerbatimOp,
        EmitC_YieldOp
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
