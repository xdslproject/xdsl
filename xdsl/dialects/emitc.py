"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

from collections.abc import Iterable, Sequence
from typing import Literal, cast

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
    IntAttrConstraint,
    IntegerAttr,
    IntegerType,
    ShapedType,
    StringAttr,
    TensorType,
    TupleType,
)
from xdsl.ir import (
    Attribute,
    AttributeCovT,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    BaseAttr,
    IRDLOperation,
    ParamAttrConstraint,
    ParsePropInAttrDict,
    get_int_constraint,
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class EmitC_ArrayType(
    ParametrizedAttribute, TypeAttribute, ShapedType, ContainerType[AttributeCovT]
):
    """EmitC array type"""

    name = "emitc.array"

    shape: ArrayAttr[IntAttr]
    element_type: AttributeCovT

    def __init__(
        self,
        shape: Iterable[int | IntAttr],
        element_type: AttributeCovT,
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__(shape, element_type)

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("EmitC array shape must not be empty")

        for dim_attr in self.shape.data:
            if dim_attr.data < 0:
                raise VerifyException(
                    "EmitC array dimensions must have non-negative size"
                )

        element_type = self.get_element_type()

        if isinstance(element_type, EmitC_ArrayType):
            raise VerifyException(
                "EmitC array element type cannot be another EmitC_ArrayType."
            )

        # Check that the element type is a supported EmitC type.
        if not self._is_valid_element_type(element_type):
            raise VerifyException(
                f"EmitC array element type '{element_type}' is not a supported EmitC type."
            )

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
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

    def _is_valid_element_type(self, element_type: Attribute) -> bool:
        """
        Check if the element type is valid for EmitC_ArrayType.
        See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L77).
        """
        return is_integer_index_or_opaque_type(
            element_type
        ) or EmitCFloatTypeConstr.verifies(element_type)


@irdl_attr_definition
class EmitC_LValueType(ParametrizedAttribute, TypeAttribute):
    """
    EmitC lvalue type.
    Values of this type can be assigned to and their address can be taken.
    See [tablegen definition](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L87)
    """

    name = "emitc.lvalue"
    value_type: TypeAttribute

    def verify(self) -> None:
        """
        Verify the LValueType.
        See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L1095)
        """
        # Check that the wrapped type is valid. This especially forbids nested lvalue types.
        if not is_supported_emitc_type(self.value_type):
            raise VerifyException(
                f"!emitc.lvalue must wrap supported emitc type, but got {self.value_type}"
            )
        if isinstance(self.value_type, EmitC_ArrayType):
            raise VerifyException("!emitc.lvalue cannot wrap !emitc.array type")


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
class EmitC_PointerType(ParametrizedAttribute, TypeAttribute):
    """EmitC pointer type"""

    name = "emitc.ptr"
    pointee_type: TypeAttribute

    def verify(self) -> None:
        if isinstance(self.pointee_type, EmitC_LValueType):
            raise VerifyException("pointers to lvalues are not allowed")


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


EmitCIntegerBitwidthConstr = get_int_constraint(Literal[1, 8, 16, 32, 64])
"""
Constraint for the bitwidth parameter of integer types supported by EmitC.
"""

EmitCIntegerConstr = ParamAttrConstraint(
    IntegerType, (IntAttrConstraint(EmitCIntegerBitwidthConstr), None)
)
"""
Constraint for integer types supported by EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
"""

EmitCFloatTypeConstr = (
    BaseAttr(Float16Type)
    | BaseAttr(BFloat16Type)
    | BaseAttr(Float32Type)
    | BaseAttr(Float64Type)
)
"""
Supported floating-point type in EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L117)
"""


def is_pointer_wide_type(type_attr: Attribute) -> bool:
    """Check if a type is a pointer-wide type."""
    match type_attr:
        case EmitC_PtrDiffT() | EmitC_SignedSizeT() | EmitC_SizeT():
            return True
        case _:
            return False


def is_integer_index_or_opaque_type(
    type_attr: Attribute,
) -> bool:
    """
    Check if a type is an integer, index, or opaque type.

    The emitC opaque type is not implemented yet so this function currently checks
    only for integer and index types.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L112).
    """
    return (
        EmitCIntegerConstr.verifies(type_attr)
        or isinstance(type_attr, IndexType)
        or is_pointer_wide_type(type_attr)
    )


def is_supported_emitc_type(type_attr: Attribute) -> bool:
    """
    Check if a type is supported by EmitC.
    See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L62).
    """

    _constrs = EmitCIntegerConstr | EmitCFloatTypeConstr
    if _constrs.verifies(type_attr):
        return True

    match type_attr:
        case IndexType():
            return True
        case EmitC_OpaqueType():
            return True
        case EmitC_ArrayType():
            elem_type = cast(Attribute, type_attr.get_element_type())
            return not isinstance(
                elem_type, EmitC_ArrayType
            ) and is_supported_emitc_type(elem_type)
        case EmitC_PointerType():
            return is_supported_emitc_type(type_attr.pointee_type)
        case TensorType():
            elem_type = cast(Attribute, type_attr.get_element_type())
            if isinstance(elem_type, EmitC_ArrayType):
                return False
            return is_supported_emitc_type(elem_type)
        case TupleType():
            return all(
                not isinstance(t, EmitC_ArrayType) and is_supported_emitc_type(t)
                for t in type_attr.types
            )
        case EmitC_PtrDiffT():
            return True
        case _:
            return False


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

    irdl_options = [ParsePropInAttrDict()]
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
                    TypeAttribute | IntegerAttr | FloatAttr,
                    # FIXME: uncomment and replace the line above when EmitC_OpaqueAttr is implemented
                    # TypeAttribute | IntegerAttr | FloatAttr | EmitC_OpaqueAttr,
                ):
                    raise VerifyException("template argument has invalid type")

        for res_type in self.res.types:
            if isinstance(res_type, EmitC_ArrayType):
                raise VerifyException("cannot return array type")


EmitC = Dialect(
    "emitc",
    [
        EmitC_CallOpaqueOp,
    ],
    [
        EmitC_ArrayType,
        EmitC_LValueType,
        EmitC_OpaqueType,
        EmitC_PointerType,
        EmitC_PtrDiffT,
        EmitC_SignedSizeT,
        EmitC_SizeT,
    ],
)
