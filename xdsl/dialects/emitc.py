from collections.abc import Iterable

from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerType,
    IntAttr,
    ShapedType,
    StringAttr,
    SymbolNameAttr,
)
from xdsl.ir import (
    Attribute,
    AttributeCovT,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.ir import (
    TypeAttribute as BuiltinTypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    BaseAttr,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.printer import Printer


@irdl_attr_definition
class EmitC_ArrayType(
    ParametrizedAttribute, TypeAttribute, ShapedType, ContainerType[AttributeCovT]
):
    """EmitC array type"""

    name = "emitc.array"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[AttributeCovT]

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int | IntAttr],
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__([shape, element_type])

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

    def print_parameters(self, printer: Printer) -> None:
        printer.print(
            "<",
            "x".join([*[str(e.data) for e in self.shape], *[str(self.element_type)]]),
        )
        printer.print(">")


@irdl_attr_definition
class EmitC_LValueType(ParametrizedAttribute, TypeAttribute):
    """EmitC lvalue type"""

    name = "emitc.lvalue"


@irdl_attr_definition
class EmitC_OpaqueType(ParametrizedAttribute, TypeAttribute):
    """EmitC opaque type"""

    name = "emitc.opaque"
    value: ParameterDef[StringAttr]


@irdl_attr_definition
class EmitC_PointerType(ParametrizedAttribute, TypeAttribute):
    """EmitC pointer type"""

    name = "emitc.ptr"
    pointee_type: ParameterDef[BuiltinTypeAttribute]

    def __init__(self, pointee_type: BuiltinTypeAttribute):
        super().__init__([pointee_type])


@irdl_attr_definition
class EmitC_PtrDiffT(ParametrizedAttribute, TypeAttribute):
    """EmitC signed pointer diff type"""

    name = "emitc.ptrdiff_t"


@irdl_attr_definition
class EmitC_SignedSizeT(ParametrizedAttribute, TypeAttribute):
    """EmitC signed size type"""

    name = "emitc.ssize_t"


@irdl_attr_definition
class EmitC_SizeT(ParametrizedAttribute, TypeAttribute):
    """EmitC unsigned size type"""

    name = "emitc.size_t"


@irdl_attr_definition
class EmitC_OpaqueAttr(ParametrizedAttribute):
    """An opaque attribute"""

    name = "emitc.opaque"
    value: ParameterDef[StringAttr]


@irdl_op_definition
class EmitC_AddOp(IRDLOperation):
    """Addition operation"""

    name = "emitc.add"
    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    v1 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_ApplyOp(IRDLOperation):
    """Apply operation"""

    name = "emitc.apply"
    assembly_format = """
        $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
      """
    applicableOperator = prop_def(AnyAttr())
    operand = operand_def(AnyOf((AnyAttr(), BaseAttr(EmitC_LValueType))))
    result = result_def(AnyAttr())


@irdl_op_definition
class EmitC_AssignOp(IRDLOperation):
    """Assign operation"""

    name = "emitc.assign"
    var = prop_def(AnyAttr())
    value = operand_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseAndOp(IRDLOperation):
    name = "emitc.bitwise_and"


@irdl_op_definition
class EmitC_BitwiseLeftShiftOp(IRDLOperation):
    name = "emitc.bitwise_left_shift"


@irdl_op_definition
class EmitC_BitwiseNotOp(IRDLOperation):
    name = "emitc.bitwise_not"


@irdl_op_definition
class EmitC_BitwiseOrOp(IRDLOperation):
    name = "emitc.bitwise_or"


@irdl_op_definition
class EmitC_BitwiseRightShiftOp(IRDLOperation):
    name = "emitc.bitwise_right_shift"


@irdl_op_definition
class EmitC_BitwiseXorOp(IRDLOperation):
    name = "emitc.bitwise_xor"


@irdl_op_definition
class EmitC_CallOp(IRDLOperation):
    name = "emitc.call"


@irdl_op_definition
class EmitC_CallOpaqueOp(IRDLOperation):
    name = "emitc.call_opaque"


@irdl_op_definition
class EmitC_CastOp(IRDLOperation):
    name = "emitc.cast"


@irdl_op_definition
class EmitC_CmpOp(IRDLOperation):
    name = "emitc.cmp"


@irdl_op_definition
class EmitC_ConditionalOp(IRDLOperation):
    name = "emitc.conditional"


@irdl_op_definition
class EmitC_ConstantOp(IRDLOperation):
    name = "emitc.constant"


@irdl_op_definition
class EmitC_DeclareFuncOp(IRDLOperation):
    name = "emitc.declare_func"


@irdl_op_definition
class EmitC_DivOp(IRDLOperation):
    name = "emitc.div"


@irdl_op_definition
class EmitC_ExpressionOp(IRDLOperation):
    name = "emitc.expression"


@irdl_op_definition
class EmitC_FileOp(IRDLOperation):
    name = "emitc.file"


@irdl_op_definition
class EmitC_ForOp(IRDLOperation):
    name = "emitc.for"


@irdl_op_definition
class EmitC_FuncOp(IRDLOperation):
    name = "emitc.func"


@irdl_op_definition
class EmitC_GetGlobalOp(IRDLOperation):
    name = "emitc.get_global"


@irdl_op_definition
class EmitC_GlobalOp(IRDLOperation):
    name = "emitc.global"
    sym_name: SymbolNameAttr = prop_def(SymbolNameAttr)
    type: EmitC_ArrayType = prop_def(EmitC_ArrayType)
    initial_value: Attribute | None = opt_prop_def(Attribute)


@irdl_op_definition
class EmitC_IfOp(IRDLOperation):
    name = "emitc.if"


@irdl_op_definition
class EmitC_IncludeOp(IRDLOperation):
    name = "emitc.include"


@irdl_op_definition
class EmitC_LiteralOp(IRDLOperation):
    name = "emitc.literal"


@irdl_op_definition
class EmitC_LoadOp(IRDLOperation):
    name = "emitc.load"


@irdl_op_definition
class EmitC_LogicalAndOp(IRDLOperation):
    name = "emitc.logical_and"


@irdl_op_definition
class EmitC_LogicalNotOp(IRDLOperation):
    name = "emitc.logical_not"


@irdl_op_definition
class EmitC_LogicalOrOp(IRDLOperation):
    name = "emitc.logical_or"


@irdl_op_definition
class EmitC_MemberOfPtrOp(IRDLOperation):
    name = "emitc.member_of_ptr"


@irdl_op_definition
class EmitC_MemberOp(IRDLOperation):
    name = "emitc.member"


@irdl_op_definition
class EmitC_MulOp(IRDLOperation):
    name = "emitc.mul"


@irdl_op_definition
class EmitC_RemOp(IRDLOperation):
    name = "emitc.rem"


@irdl_op_definition
class EmitC_ReturnOp(IRDLOperation):
    name = "emitc.return"


@irdl_op_definition
class EmitC_SubOp(IRDLOperation):
    name = "emitc.sub"


@irdl_op_definition
class EmitC_SubscriptOp(IRDLOperation):
    name = "emitc.subscript"


@irdl_op_definition
class EmitC_SwitchOp(IRDLOperation):
    name = "emitc.switch"


@irdl_op_definition
class EmitC_UnaryMinusOp(IRDLOperation):
    name = "emitc.unary_minus"


@irdl_op_definition
class EmitC_UnaryPlusOp(IRDLOperation):
    name = "emitc.unary_plus"


@irdl_op_definition
class EmitC_VariableOp(IRDLOperation):
    name = "emitc.variable"


@irdl_op_definition
class EmitC_VerbatimOp(IRDLOperation):
    name = "emitc.verbatim"


@irdl_op_definition
class EmitC_YieldOp(IRDLOperation):
    name = "emitc.yield"


EmitC = Dialect(
    "emitc",
    [
        EmitC_AddOp,
        EmitC_ApplyOp,
        EmitC_AssignOp,
        EmitC_BitwiseAndOp,
        EmitC_BitwiseLeftShiftOp,
        EmitC_BitwiseNotOp,
        EmitC_BitwiseOrOp,
        EmitC_BitwiseRightShiftOp,
        EmitC_BitwiseXorOp,
        EmitC_CallOp,
        EmitC_CallOpaqueOp,
        EmitC_CastOp,
        EmitC_CmpOp,
        EmitC_ConditionalOp,
        EmitC_ConstantOp,
        EmitC_DeclareFuncOp,
        EmitC_DivOp,
        EmitC_ExpressionOp,
        EmitC_FileOp,
        EmitC_ForOp,
        EmitC_FuncOp,
        EmitC_GetGlobalOp,
        EmitC_GlobalOp,
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
        EmitC_ReturnOp,
        EmitC_SubOp,
        EmitC_SubscriptOp,
        EmitC_SwitchOp,
        EmitC_UnaryMinusOp,
        EmitC_UnaryPlusOp,
        EmitC_VariableOp,
        EmitC_VerbatimOp,
        EmitC_YieldOp,
    ],
    [
        EmitC_ArrayType,
        EmitC_LValueType,
        EmitC_OpaqueType,
        EmitC_PointerType,
        EmitC_PtrDiffT,
        EmitC_SignedSizeT,
        EmitC_SizeT,
        EmitC_OpaqueAttr,
    ],
)
