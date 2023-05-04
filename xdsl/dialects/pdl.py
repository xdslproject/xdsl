from __future__ import annotations

from typing import Annotated, Generic, Sequence, TypeVar

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    TypeAttribute,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    OpAttr,
    Operand,
    OptOpAttr,
    OptOperand,
    OptRegion,
    ParameterDef,
    VarOpResult,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class AttributeType(ParametrizedAttribute, TypeAttribute):
    name = "pdl.attribute"


@irdl_attr_definition
class OperationType(ParametrizedAttribute, TypeAttribute):
    name = "pdl.operation"


@irdl_attr_definition
class TypeType(ParametrizedAttribute, TypeAttribute):
    name = "pdl.type"


@irdl_attr_definition
class ValueType(ParametrizedAttribute, TypeAttribute):
    name = "pdl.value"


AnyPDLType = AttributeType | OperationType | TypeType | ValueType

_RangeT = TypeVar(
    "_RangeT",
    bound=AttributeType | OperationType | TypeType | ValueType,
    covariant=True,
)


@irdl_attr_definition
class RangeType(Generic[_RangeT], ParametrizedAttribute, TypeAttribute):
    name = "pdl.range"
    elementType: ParameterDef[_RangeT]


@irdl_op_definition
class ApplyNativeConstraintOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_constraint-mlirpdlapplynativeconstraintop
    """

    name: str = "pdl.apply_native_constraint"
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]

    def verify_(self) -> None:
        if "name" not in self.attributes:
            raise VerifyException("ApplyNativeConstraintOp requires a 'name' attribute")

        if not isinstance(self.attributes["name"], StringAttr):
            raise VerifyException("expected 'name' attribute to be a StringAttr")

    @staticmethod
    def get(name: str, args: Sequence[SSAValue]) -> ApplyNativeConstraintOp:
        return ApplyNativeConstraintOp.build(
            result_types=[], operands=[args], attributes={"name": StringAttr(name)}
        )


@irdl_op_definition
class ApplyNativeRewriteOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_rewrite-mlirpdlapplynativerewriteop
    """

    name: str = "pdl.apply_native_rewrite"
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]
    res: Annotated[VarOpResult, AnyPDLType]

    def verify_(self) -> None:
        if "name" not in self.attributes:
            raise VerifyException("ApplyNativeRewriteOp requires a 'name' attribute")

        if not isinstance(self.attributes["name"], StringAttr):
            raise VerifyException("expected 'name' attribute to be a StringAttr")

    @staticmethod
    def get(
        name: str, args: Sequence[SSAValue], result_types: Sequence[Attribute]
    ) -> ApplyNativeRewriteOp:
        return ApplyNativeRewriteOp.build(
            result_types=[result_types],
            operands=[args],
            attributes={"name": StringAttr(name)},
        )


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
    """

    name: str = "pdl.attribute"
    value: OptOpAttr[Attribute]
    valueType: Annotated[OptOperand, TypeType]
    output: Annotated[OpResult, AttributeType]

    @staticmethod
    def get(
        value: Attribute | None = None, valueType: SSAValue | None = None
    ) -> AttributeOp:
        attributes: dict[str, Attribute] = {}
        if value is not None:
            attributes["value"] = value

        if valueType is None:
            value_type = []
        else:
            value_type = [valueType]

        return AttributeOp.build(
            operands=[value_type], attributes=attributes, result_types=[AttributeType()]
        )


@irdl_op_definition
class EraseOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-mlirpdleraseop
    """

    name: str = "pdl.erase"
    opValue: Annotated[Operand, OperationType]

    @staticmethod
    def get(opValue: SSAValue) -> EraseOp:
        return EraseOp.build(operands=[opValue], result_types=[])


@irdl_op_definition
class OperandOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-mlirpdloperandop
    """

    name: str = "pdl.operand"
    valueType: Annotated[OptOperand, TypeType]
    value: Annotated[OpResult, ValueType]

    @staticmethod
    def get(valueType: SSAValue | None = None) -> OperandOp:
        if valueType is None:
            value_type = []
        else:
            value_type = [valueType]
        return OperandOp.build(operands=[value_type], result_types=[ValueType()])


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-mlirpdloperandsop
    """

    name: str = "pdl.operands"
    valueType: Annotated[Operand, RangeType[TypeType]]
    value: Annotated[OpResult, RangeType[ValueType]]


@irdl_op_definition
class OperationOp(IRDLOperation):
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

    @staticmethod
    def get(
        opName: StringAttr | None,
        attributeValueNames: ArrayAttr[StringAttr] | None = None,
        operandValues: Sequence[SSAValue] | None = None,
        attributeValues: Sequence[SSAValue] | None = None,
        typeValues: Sequence[SSAValue] | None = None,
    ):
        if attributeValueNames is None:
            attributeValueNames = ArrayAttr([])
        if operandValues is None:
            operandValues = []
        if attributeValues is None:
            attributeValues = []
        if typeValues is None:
            typeValues = []

        return OperationOp.build(
            operands=[operandValues, attributeValues, typeValues],
            result_types=[OperationType()],
            attributes={"attributeValueNames": attributeValueNames, "opName": opName},
        )


@irdl_op_definition
class PatternOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlpattern-mlirpdlpatternop
    """

    name: str = "pdl.pattern"
    benefit: OpAttr[IntegerAttr[IntegerType]]
    sym_name: OptOpAttr[StringAttr]
    body: Region

    @staticmethod
    def get(
        benefit: IntegerAttr[IntegerType], sym_name: StringAttr | None, body: Region
    ) -> PatternOp:
        return PatternOp.build(
            attributes={
                "benefit": benefit,
                "sym_name": sym_name,
            },
            regions=[body],
            result_types=[],
        )

    @staticmethod
    def from_callable(
        benefit: IntegerAttr[IntegerType],
        sym_name: StringAttr | None,
        callable: Block.BlockCallback,
    ) -> PatternOp:
        block = Block.from_callable([], callable)
        region = Region(block)
        return PatternOp.get(benefit, sym_name, region)


@irdl_op_definition
class RangeOp(IRDLOperation):
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
                          {elem_type}, current element type: {cur_elem_type}"
                    )


@irdl_op_definition
class ReplaceOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlreplace-mlirpdlreplaceop

    pdl.replace` operations are used within `pdl.rewrite` regions to specify
    that an input operation should be marked as replaced. The semantics of this
    operation correspond with the `replaceOp` method on a `PatternRewriter`. The
    set of replacement values can be either:
    * a single `Operation` (`replOperation` should be populated)
      - The operation will be replaced with the results of this operation.
    * a set of `Value`s (`replValues` should be populated)
      - The operation will be replaced with these values.
    """

    name: str = "pdl.replace"
    opValue: Annotated[Operand, OperationType]
    replOperation: Annotated[OptOperand, OperationType]
    replValues: Annotated[VarOperand, ValueType | ArrayAttr[ValueType]]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        opValue: SSAValue,
        replOperation: SSAValue | None = None,
        replValues: Sequence[SSAValue] | None = None,
    ) -> ReplaceOp:
        operands: list[SSAValue | Sequence[SSAValue]] = [opValue]
        if replOperation is None:
            operands.append([])
        else:
            operands.append([replOperation])
        if replValues is None:
            replValues = []
        operands.append(replValues)
        return ReplaceOp.build(operands=operands)

    def verify_(self) -> None:
        if self.replOperation is None:
            if not len(self.replValues):
                raise VerifyException(
                    "Exactly one of `replOperation` or "
                    "`replValues` must be set in `ReplaceOp`"
                    ", both are empty"
                )
        elif len(self.replValues):
            raise VerifyException(
                "Exactly one of `replOperation` or `replValues` must be set in "
                "`ReplaceOp`, both are set"
            )


@irdl_op_definition
class ResultOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresult-mlirpdlresultop
    """

    name: str = "pdl.result"
    index: OpAttr[IntegerAttr[IntegerType]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType]

    @staticmethod
    def get(index: IntegerAttr[IntegerType], parent: SSAValue) -> ResultOp:
        return ResultOp.build(
            operands=[parent], attributes={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresults-mlirpdlresultsop
    """

    name: str = "pdl.results"
    index: OpAttr[IntegerAttr[IntegerType]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType | ArrayAttr[ValueType]]


@irdl_op_definition
class RewriteOp(IRDLOperation):
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
        if "name" in self.attributes:
            if not isinstance(self.attributes["name"], StringAttr):
                raise Exception("expected 'name' attribute to be a StringAttr")

    @staticmethod
    def get(
        name: StringAttr | None,
        root: SSAValue | None,
        external_args: Sequence[SSAValue],
        body: Region | None,
    ) -> RewriteOp:
        operands: list[SSAValue | Sequence[SSAValue]] = []
        if root is not None:
            operands.append([root])
        else:
            operands.append([])
        operands.append(external_args)

        regions: list[Region | list[Region]] = []
        if body is not None:
            regions.append([body])
        else:
            regions.append([])

        attributes: dict[str, Attribute] = {}
        if name is not None:
            attributes["name"] = name

        return RewriteOp.build(
            result_types=[], operands=operands, attributes=attributes, regions=regions
        )

    @staticmethod
    def from_callable(
        name: StringAttr | None,
        root: SSAValue | None,
        external_args: Sequence[SSAValue],
        body: Block.BlockCallback,
    ) -> RewriteOp:
        block = Block.from_callable([], body)
        region = Region(block)
        return RewriteOp.get(name, root, external_args, region)


@irdl_op_definition
class TypeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-mlirpdltypeop
    """

    name: str = "pdl.type"
    constantType: OptOpAttr[Attribute]
    result: Annotated[OpResult, TypeType]

    @staticmethod
    def get(constantType: TypeType | None = None) -> TypeOp:
        return TypeOp.build(
            attributes={"constantType": constantType}, result_types=[TypeType()]
        )


@irdl_op_definition
class TypesOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltypes-mlirpdltypesop
    """

    name: str = "pdl.types"
    constantTypes: Annotated[OptOperand, ArrayAttr[TypeType]]
    result: Annotated[OpResult, RangeType[TypeType]]


PDL = Dialect(
    [
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
    ],
    [
        AttributeType,
        OperationType,
        TypeType,
        ValueType,
        RangeType,
    ],
)
