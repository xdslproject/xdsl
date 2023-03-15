from __future__ import annotations

from typing import Annotated, Generic, TypeVar

from xdsl.dialects.builtin import (ArrayAttr, IntegerAttr, IntegerType,
                                   StringAttr)
from xdsl.ir import (Attribute, Dialect, MLIRType, OpResult, Operation,
                     ParametrizedAttribute, Region, SSAValue)
from xdsl.irdl import (AttrSizedOperandSegments, OpAttr, Operand, OptOpAttr,
                       OptOperand, OptRegion, ParameterDef, VarOpResult,
                       VarOperand, irdl_attr_definition, irdl_op_definition)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class AttributeType(ParametrizedAttribute, MLIRType):
    name = "pdl.attribute"


@irdl_attr_definition
class OperationType(ParametrizedAttribute, MLIRType):
    name = "pdl.operation"


@irdl_attr_definition
class TypeType(ParametrizedAttribute, MLIRType):
    name = "pdl.type"


@irdl_attr_definition
class ValueType(ParametrizedAttribute, MLIRType):
    name = "pdl.value"


AnyPDLType = AttributeType | OperationType | TypeType | ValueType

_RangeT = TypeVar("_RangeT",
                  bound=AttributeType | OperationType | TypeType | ValueType,
                  covariant=True)


@irdl_attr_definition
class RangeType(Generic[_RangeT], ParametrizedAttribute, MLIRType):
    name = "pdl.range"
    elementType: ParameterDef[_RangeT]


@irdl_op_definition
class ApplyNativeConstraintOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_constraint-mlirpdlapplynativeconstraintop
    """
    name: str = "pdl.apply_native_constraint"
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]

    def verify_(self) -> None:
        if 'name' not in self.attributes:
            raise VerifyException(
                "ApplyNativeConstraintOp requires a 'name' attribute")

        if not isinstance(self.attributes['name'], StringAttr):
            raise VerifyException(
                "expected 'name' attribute to be a StringAttr")

    @staticmethod
    def get(name: str, args: list[SSAValue]) -> ApplyNativeConstraintOp:
        return ApplyNativeConstraintOp.build(
            result_types=[],
            operands=[args],
            attributes={"name": StringAttr(name)})


@irdl_op_definition
class ApplyNativeRewriteOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_rewrite-mlirpdlapplynativerewriteop
    """
    name: str = "pdl.apply_native_rewrite"
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]
    res: Annotated[VarOpResult, AnyPDLType]

    def verify_(self) -> None:
        if 'name' not in self.attributes:
            raise VerifyException(
                "ApplyNativeRewriteOp requires a 'name' attribute")

        if not isinstance(self.attributes['name'], StringAttr):
            raise VerifyException(
                "expected 'name' attribute to be a StringAttr")

    @staticmethod
    def get(name: str, args: list[SSAValue],
            result_types: list[Attribute]) -> ApplyNativeRewriteOp:

        return ApplyNativeRewriteOp.build(
            result_types=[result_types],
            operands=[args],
            attributes={"name": StringAttr(name)})


@irdl_op_definition
class AttributeOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
    """
    name: str = "pdl.attribute"
    value: OptOpAttr[Attribute]
    valueType: Annotated[OptOperand, TypeType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class EraseOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-mlirpdleraseop
    """
    name: str = "pdl.erase"
    opValue: Annotated[Operand, OperationType]


@irdl_op_definition
class OperandOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-mlirpdloperandop
    """
    name: str = "pdl.operand"
    valueType: Annotated[OptOperand, TypeType]
    value: Annotated[OpResult, ValueType]


@irdl_op_definition
class OperandsOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-mlirpdloperandsop
    """
    name: str = "pdl.operands"
    valueType: Annotated[Operand, RangeType[TypeType]]
    value: Annotated[OpResult, RangeType[ValueType]]


@irdl_op_definition
class OperationOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperation-mlirpdloperationop
    """
    name: str = "pdl.operation"
    opName: OptOpAttr[StringAttr]
    attributeValueNames: OpAttr[ArrayAttr[StringAttr]]

    operandValues: Annotated[VarOperand, ValueType | RangeType[ValueType]]
    attributeValues: Annotated[VarOperand, AttributeType]
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
            if isa(arg.typ, RangeType[AnyPDLType]):
                return arg.typ.elementType
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
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OptOpAttr[StringAttr]
    # parameters of external rewriter function
    externalArgs: Annotated[VarOperand, AnyPDLType]
    # body of inline rewriter function
    body: OptRegion

    irdl_options = [AttrSizedOperandSegments()]

    def verify_(self) -> None:
        if 'name' in self.attributes:
            if not isinstance(self.attributes['name'], StringAttr):
                raise Exception("expected 'name' attribute to be a StringAttr")

    @staticmethod
    def get(name: str, root: SSAValue | None, external_args: list[SSAValue],
            body: Region | None) -> RewriteOp:

        operands: list[SSAValue | list[SSAValue]] = []
        if root is not None:
            operands.append([root])
        else:
            operands.append([])
        operands.append(external_args)

        regions: list[Region | list[Region]] = []
        if body is not None:
            regions.append(body)
        else:
            regions.append([])

        return RewriteOp.build(result_types=[],
                               operands=operands,
                               attributes={"name": StringAttr(name)},
                               regions=regions)


@irdl_op_definition
class TypeOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-mlirpdltypeop
    """
    name: str = "pdl.type"
    constantType: Annotated[OptOperand, TypeType]
    result: Annotated[OpResult, TypeType]


@irdl_op_definition
class TypesOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltypes-mlirpdltypesop
    """
    name: str = "pdl.types"
    constantTypes: Annotated[OptOperand, ArrayAttr[TypeType]]
    result: Annotated[OpResult, RangeType[TypeType]]


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
