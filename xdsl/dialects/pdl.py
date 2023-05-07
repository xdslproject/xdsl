from __future__ import annotations

from typing import Annotated, Generic, Iterable, Sequence, TypeVar

from xdsl.dialects.builtin import (
    AnyArrayAttr,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    i32,
)
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
    element_type: ParameterDef[_RangeT]

    def __init__(self, element_type: _RangeT):
        super().__init__([element_type])


@irdl_op_definition
class ApplyNativeConstraintOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_constraint-mlirpdlapplynativeconstraintop
    """

    name = "pdl.apply_native_constraint"
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]

    @property
    def constraint_name(self) -> StringAttr:
        name = self.attributes.get("name", None)
        if not isinstance(name, StringAttr):
            raise VerifyException(
                f"Operation {self.name} requires a StringAttr 'name' attribute"
            )
        return name

    @constraint_name.setter
    def constraint_name(self, name: StringAttr) -> None:
        self.attributes["name"] = name

    def verify_(self) -> None:
        if "name" not in self.attributes:
            raise VerifyException("ApplyNativeConstraintOp requires a 'name' attribute")

        if not isinstance(self.attributes["name"], StringAttr):
            raise VerifyException("expected 'name' attribute to be a StringAttr")

    def __init__(self, name: str | StringAttr, args: Sequence[SSAValue]) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(operands=[args], attributes={"name": name})


@irdl_op_definition
class ApplyNativeRewriteOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_rewrite-mlirpdlapplynativerewriteop
    """

    name = "pdl.apply_native_rewrite"
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OpAttr[StringAttr]
    args: Annotated[VarOperand, AnyPDLType]
    res: Annotated[VarOpResult, AnyPDLType]

    @property
    def constraint_name(self) -> StringAttr:
        name = self.attributes.get("name", None)
        if not isinstance(name, StringAttr):
            raise VerifyException(
                f"Operation {self.name} requires a StringAttr 'name' attribute"
            )
        return name

    @constraint_name.setter
    def constraint_name(self, name: StringAttr) -> None:
        self.attributes["name"] = name

    def __init__(
        self,
        name: str | StringAttr,
        args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(
            result_types=[result_types],
            operands=[args],
            attributes={"name": name},
        )


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
    """

    name = "pdl.attribute"
    value: OptOpAttr[Attribute]
    value_type: Annotated[OptOperand, TypeType]
    output: Annotated[OpResult, AttributeType]

    def verify_(self):
        if self.value is not None and self.value_type is not None:
            raise VerifyException(
                f"{self.name} cannot both specify an expected attribute "
                "via a constant value and an expected type."
            )

    def __init__(self, value: Attribute | SSAValue | None = None) -> None:
        """
        The given value is either the expected attribute, if given an attribute, or the
        expected attribute type, if given an SSAValue.
        """
        attributes: dict[str, Attribute] = {}
        operands: list[SSAValue | None] = [None]
        if isinstance(value, Attribute):
            attributes["value"] = value
        elif isinstance(value, SSAValue):
            operands = [value]

        super().__init__(
            operands=operands, attributes=attributes, result_types=[AttributeType()]
        )


@irdl_op_definition
class EraseOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-mlirpdleraseop
    """

    name = "pdl.erase"
    op_value: Annotated[Operand, OperationType]

    def __init__(self, op_value: SSAValue) -> None:
        super().__init__(operands=[op_value])


@irdl_op_definition
class OperandOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-mlirpdloperandop
    """

    name = "pdl.operand"
    value_type: Annotated[OptOperand, TypeType]
    value: Annotated[OpResult, ValueType]

    def __init__(self, value_type: SSAValue | None = None) -> None:
        super().__init__(operands=[value_type], result_types=[ValueType()])


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-mlirpdloperandsop
    """

    name = "pdl.operands"
    value_type: Annotated[OptOperand, RangeType[TypeType]]
    value: Annotated[OpResult, RangeType[ValueType]]

    def __init__(self, value_type: SSAValue | None) -> None:
        super().__init__(operands=[value_type], result_types=[RangeType(ValueType())])


@irdl_op_definition
class OperationOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperation-mlirpdloperationop
    """

    name = "pdl.operation"
    opName: OptOpAttr[StringAttr]
    attributeValueNames: OpAttr[ArrayAttr[StringAttr]]

    operand_values: Annotated[VarOperand, ValueType | RangeType[ValueType]]
    attribute_values: Annotated[VarOperand, AttributeType]
    type_values: Annotated[VarOperand, TypeType | RangeType[TypeType]]
    op: Annotated[OpResult, OperationType]

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        op_name: str | StringAttr | None,
        attribute_value_names: ArrayAttr[StringAttr] | None = None,
        operand_values: Sequence[SSAValue] | None = None,
        attribute_values: Sequence[SSAValue] | None = None,
        type_values: Sequence[SSAValue] | None = None,
    ):
        if isinstance(op_name, str):
            op_name = StringAttr(op_name)
        if attribute_value_names is None:
            attribute_value_names = ArrayAttr([])
        if operand_values is None:
            operand_values = []
        if attribute_values is None:
            attribute_values = []
        if type_values is None:
            type_values = []

        super().__init__(
            operands=[operand_values, attribute_values, type_values],
            result_types=[OperationType()],
            attributes={
                "attributeValueNames": attribute_value_names,
                "opName": op_name,
            },
        )


@irdl_op_definition
class PatternOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlpattern-mlirpdlpatternop
    """

    name = "pdl.pattern"
    benefit: OpAttr[IntegerAttr[Annotated[IntegerType, IntegerType(16)]]]
    sym_name: OptOpAttr[StringAttr]
    body: Region

    def __init__(
        self,
        benefit: int | IntegerAttr[IntegerType],
        sym_name: str | StringAttr | None,
        body: Region | Block.BlockCallback,
    ):
        if isinstance(benefit, int):
            benefit = IntegerAttr(benefit, 16)
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if not isinstance(body, Region):
            body = Region(Block.from_callable([], body))
        super().__init__(
            attributes={
                "benefit": benefit,
                "sym_name": sym_name,
            },
            regions=[body],
            result_types=[],
        )


@irdl_op_definition
class RangeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrange-mlirpdlrangeop
    """

    name = "pdl.range"
    arguments: Annotated[VarOperand, AnyPDLType | RangeType[AnyPDLType]]
    result: Annotated[OpResult, RangeType[AnyPDLType]]

    def verify_(self) -> None:
        def get_type_or_elem_type(arg: SSAValue) -> Attribute:
            if isa(arg.typ, RangeType[AnyPDLType]):
                return arg.typ.element_type
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

    def __init__(
        self,
        arguments: Sequence[SSAValue],
        result_type: RangeType[AnyPDLType] | None = None,
    ) -> None:
        if result_type is None:
            if len(arguments) == 0:
                raise ValueError("Empty range constructions require a return type.")

            if isa(arguments[0].typ, RangeType[AnyPDLType]):
                result_type = RangeType(arguments[0].typ.element_type)
            elif isa(arguments[0].typ, AnyPDLType):
                result_type = RangeType(arguments[0].typ)
            else:
                raise ValueError(
                    f"Arguments of {self.name} are expected to be PDL types"
                )

        super().__init__(operands=arguments, result_types=[result_type])


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

    name = "pdl.replace"
    op_value: Annotated[Operand, OperationType]
    repl_operation: Annotated[OptOperand, OperationType]
    repl_values: Annotated[VarOperand, ValueType | ArrayAttr[ValueType]]

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        op_value: SSAValue,
        repl_operation: SSAValue | None = None,
        repl_values: Sequence[SSAValue] | None = None,
    ) -> None:
        operands: list[SSAValue | Sequence[SSAValue]] = [op_value]
        if repl_operation is None:
            operands.append([])
        else:
            operands.append([repl_operation])
        if repl_values is None:
            repl_values = []
        operands.append(repl_values)
        super().__init__(operands=operands)

    def verify_(self) -> None:
        if self.repl_operation is None:
            if not len(self.repl_values):
                raise VerifyException(
                    "Exactly one of `replOperation` or "
                    "`replValues` must be set in `ReplaceOp`"
                    ", both are empty"
                )
        elif len(self.repl_values):
            raise VerifyException(
                "Exactly one of `replOperation` or `replValues` must be set in "
                "`ReplaceOp`, both are set"
            )


@irdl_op_definition
class ResultOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresult-mlirpdlresultop
    """

    name = "pdl.result"
    index: OpAttr[IntegerAttr[Annotated[IntegerType, i32]]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType]

    def __init__(self, index: int | IntegerAttr[IntegerType], parent: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, 32)
        super().__init__(
            operands=[parent], attributes={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresults-mlirpdlresultsop
    """

    name = "pdl.results"
    index: OpAttr[IntegerAttr[Annotated[IntegerType, i32]]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType | ArrayAttr[ValueType]]

    def __init__(
        self,
        index: int | IntegerAttr[IntegerType],
        parent: SSAValue,
        result_type: ValueType | ArrayAttr[ValueType],
    ) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, 32)
        super().__init__(
            operands=[parent], result_types=[result_type], attributes={"index": index}
        )


@irdl_op_definition
class RewriteOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrewrite-mlirpdlrewriteop
    """

    name = "pdl.rewrite"
    root: Annotated[OptOperand, OperationType]
    # name of external rewriter function
    # https://github.com/xdslproject/xdsl/issues/98
    # name: OptOpAttr[StringAttr]
    # parameters of external rewriter function
    external_args: Annotated[VarOperand, AnyPDLType]
    # body of inline rewriter function
    body: OptRegion

    irdl_options = [AttrSizedOperandSegments()]

    def verify_(self) -> None:
        if "name" in self.attributes:
            if not isinstance(self.attributes["name"], StringAttr):
                raise Exception("expected 'name' attribute to be a StringAttr")

    def __init__(
        self,
        name: str | StringAttr | None,
        root: SSAValue | None,
        external_args: Sequence[SSAValue],
        body: Region | Block.BlockCallback | None,
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)

        operands: list[SSAValue | Sequence[SSAValue]] = []
        if root is not None:
            operands.append([root])
        else:
            operands.append([])
        operands.append(external_args)

        regions: list[Region | list[Region]] = []
        if isinstance(body, Region):
            regions.append([body])
        elif body is not None:
            regions.append(Region(Block.from_callable([], body)))
        else:
            regions.append([])

        attributes: dict[str, Attribute] = {}
        if name is not None:
            attributes["name"] = name

        super().__init__(
            result_types=[],
            operands=operands,
            attributes=attributes,
            regions=regions,
        )


@irdl_op_definition
class TypeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-mlirpdltypeop
    """

    name = "pdl.type"
    constantType: OptOpAttr[Attribute]
    result: Annotated[OpResult, TypeType]

    def __init__(self, constant_type: TypeType | None = None) -> None:
        super().__init__(
            attributes={"constantType": constant_type}, result_types=[TypeType()]
        )


@irdl_op_definition
class TypesOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltypes-mlirpdltypesop
    """

    name = "pdl.types"
    constantTypes: OptOpAttr[AnyArrayAttr]
    result: Annotated[OpResult, RangeType[TypeType]]

    def __init__(self, constant_types: Iterable[TypeType] = ()) -> None:
        super().__init__(
            attributes={"constantType": ArrayAttr(constant_types)},
            result_types=[RangeType(TypeType())],
        )


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
