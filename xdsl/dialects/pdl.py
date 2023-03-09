from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *


@irdl_attr_definition
class AttributeType(ParametrizedAttribute):
    name = "pdl.attribute"


@irdl_attr_definition
class OperationType(ParametrizedAttribute):
    name = "pdl.operation"


@irdl_attr_definition
class TypeType(ParametrizedAttribute):
    name = "pdl.type"


@irdl_attr_definition
class ValueType(ParametrizedAttribute):
    name = "pdl.value"


AnyPDLType = AttributeType | OperationType | TypeType | ValueType

_RangeT = TypeVar("_RangeT",
                  bound=AttributeType | OperationType | TypeType | ValueType,
                  covariant=True)


@irdl_attr_definition
class RangeType(Generic[_RangeT], ParametrizedAttribute):
    name = "pdl.range"
    element_type: ParameterDef[_RangeT]


@irdl_op_definition
class ApplyNativeConstraintOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_constraint-mlirpdlapplynativeconstraintop
    """
    name: str = "pdl.apply_native_constraint"
    name_: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]


@irdl_op_definition
class ApplyNativeRewriteOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_rewrite-mlirpdlapplynativerewriteop
    """
    name: str = "pdl.apply_native_rewrite"
    name_: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]
    results: Annotated[VarOpResult, AnyPDLType]


@irdl_op_definition
class AttributeOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
    """
    name: str = "pdl.attribute"
    value: OptOpAttr[Attribute]
    value_type: Annotated[OptOperand, TypeType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class EraseOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-mlirpdleraseop
    """
    name: str = "pdl.erase"
    op_value: Annotated[Operand, OperationType]


@irdl_op_definition
class OperandOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-mlirpdloperandop
    """
    name: str = "pdl.operand"
    value_type: Annotated[OptOperand, TypeType]
    output: Annotated[OpResult, ValueType]


@irdl_op_definition
class OperandsOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-mlirpdloperandsop
    """
    name: str = "pdl.operands"
    value_type: Annotated[Operand, RangeType[TypeType]]
    output: Annotated[OpResult, RangeType[ValueType]]


@irdl_op_definition
class OperationOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperation-mlirpdloperationop
    """
    name: str = "pdl.operation"
    opName: OptOpAttr[StringAttr]
    attributeValueNames: OpAttr[ArrayAttr[StringAttr]]

    operandValues: Annotated[VarOperand, ValueType | RangeType[ValueType]]
    # in PDL docs, this is just a handle to AttributeType, not a range.
    # Why is it different to operandvalues
    attributeValues: Annotated[VarOperand,
                               AttributeType | RangeType[AttributeType]]
    typeValues: Annotated[VarOperand, TypeType | RangeType[TypeType]]
    op: Annotated[OpResult, OperationType]

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class PatternOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlpattern-mlirpdlpatternop
    """
    name: str = "pdl.pattern"
    benefit: OpAttr[IntegerAttr[IntegerType]]
    sym_name: OpAttr[StringAttr]
    body: Region


@irdl_op_definition
class RangeOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrange-mlirpdlrangeop
    """
    name: str = "pdl.range"
    arguments: Annotated[VarOperand, AnyPDLType | RangeType[AnyPDLType]]
    result: Annotated[OpResult, RangeType[AnyPDLType]]

    def verify_(self) -> None:

        def get_type_or_elem_type(arg: SSAValue) -> Attribute:
            if isinstance(arg.typ, RangeType):
                return arg.typ.element_type  # type: ignore
            else:
                return arg.typ

        if len(self.arguments) > 0:
            elem_type = get_type_or_elem_type(self.result)

            for arg in self.arguments:
                if cur_elem_type := get_type_or_elem_type(arg) != elem_type:
                    raise VerifyException(
                        f"All arguments must have the same type or be an array  \
                          of the corresponding element type. First element type:\
                          {elem_type}, current element type: {cur_elem_type}")


@irdl_op_definition
class ReplaceOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlreplace-mlirpdlreplaceop
    """
    name: str = "pdl.replace"
    opValue: Annotated[Operand, OperationType]
    replOperation: Annotated[OptOperand, OperationType]
    replValues: Annotated[VarOperand, ValueType | ArrayAttr[ValueType]]

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class ResultOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresult-mlirpdlresultop
    """
    name: str = "pdl.result"
    index: OpAttr[IntegerAttr[IntegerType]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType]


@irdl_op_definition
class ResultsOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresults-mlirpdlresultsop
    """
    name: str = "pdl.results"
    index: OpAttr[IntegerAttr[IntegerType]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType | ArrayAttr[ValueType]]


@irdl_op_definition
class RewriteOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrewrite-mlirpdlrewriteop
    """
    name: str = "pdl.rewrite"
    root: Annotated[OptOperand, OperationType]
    # name of external rewriter function
    name_: OptOpAttr[StringAttr]
    externalArgs: Annotated[VarOperand, AnyPDLType]
    body: OptRegion

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class TypeOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-mlirpdltypeop
    """
    name: str = "pdl.type"
    constant_type: Annotated[OptOperand, TypeType]
    result: Annotated[OpResult, TypeType]


@irdl_op_definition
class TypesOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltypes-mlirpdltypesop
    """
    name: str = "pdl.types"
    constant_types: Annotated[OptOperand, ArrayAttr[TypeType]]
    result: Annotated[OpResult, ArrayAttr[TypeType]]


PDL = Dialect([
    ApplyNativeConstraintOp,
    ApplyNativeRewriteOp,
    AttributeOp,
    OperandOp,
    EraseOp,
    OperandsOp,
    OperationOp,
    PatternOp,
    RangeOp,
    ReplaceOp,
    ResultOp,
    ResultsOp,
    RewriteOp,
    TypeOp,
    TypesOp,
], [
    AttributeType,
    OperationType,
    TypeType,
    ValueType,
    RangeType,
])
