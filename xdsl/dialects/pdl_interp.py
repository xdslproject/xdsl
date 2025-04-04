from __future__ import annotations

from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass
from typing import ClassVar

from xdsl.dialects.builtin import (
    I16,
    I32,
    ArrayAttr,
    ContainerOf,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
)
from xdsl.dialects.pdl import (
    AnyPDLTypeConstr,
    AttributeType,
    OperationType,
    RangeType,
    TypeType,
    ValueType,
)
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    AttrSizedOperandSegments,
    ConstraintContext,
    GenericAttrConstraint,
    IRDLOperation,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    CallableOpInterface,
    IsolatedFromAbove,
    IsTerminator,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

boolLike = ContainerOf(IntegerType(1))
signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))


@irdl_op_definition
class GetOperandOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_operand-pdl_interpgetoperandop
    """

    name = "pdl_interp.get_operand"
    index = prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType)

    assembly_format = "$index `of` $input_op attr-dict"

    def __init__(self, index: int | IntegerAttr[I32], input_op: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            operands=[input_op], properties={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class FinalizeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfinalize-pdl_interpfinalizeop
    """

    name = "pdl_interp.finalize"
    traits = traits_def(IsTerminator())

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class CheckOperationNameOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_operation_name-pdl_interpcheckoperationnameop
    """

    name = "pdl_interp.check_operation_name"
    traits = traits_def(IsTerminator())
    operation_name = prop_def(StringAttr, prop_name="name")
    input_op = operand_def(OperationType)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = (
        "`of` $input_op `is` $name attr-dict `->` $true_dest `, ` $false_dest"
    )

    def __init__(
        self,
        operation_name: str | StringAttr,
        input_op: SSAValue,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        if isinstance(operation_name, str):
            operation_name = StringAttr(operation_name)
        super().__init__(
            operands=[input_op],
            properties={"name": operation_name},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class CheckOperandCountOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_operand_count-pdl_interpcheckoperandcountop
    """

    name = "pdl_interp.check_operand_count"
    traits = traits_def(IsTerminator())
    input_op = operand_def(OperationType)
    count = prop_def(IntegerAttr[I32])
    compareAtLeast = opt_prop_def(UnitAttr)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = "`of` $input_op `is` (`at_least` $compareAtLeast^)? $count attr-dict `->` $true_dest `, ` $false_dest"

    def __init__(
        self,
        input_op: SSAValue,
        count: int | IntegerAttr[I32],
        trueDest: Block,
        falseDest: Block,
        compareAtLeast: bool = False,
    ) -> None:
        if isinstance(count, int):
            count = IntegerAttr.from_int_and_width(count, 32)
        properties = dict[str, Attribute](count=count)
        if compareAtLeast:
            properties["compareAtLeast"] = UnitAttr()
        super().__init__(
            operands=[input_op],
            properties=properties,
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class CheckResultCountOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_result_count-pdl_interpcheckresultcountop
    """

    name = "pdl_interp.check_result_count"
    traits = traits_def(IsTerminator())
    input_op = operand_def(OperationType)
    count = prop_def(IntegerAttr[I32])
    compareAtLeast = opt_prop_def(UnitAttr)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = "`of` $input_op `is` (`at_least` $compareAtLeast^)? $count attr-dict `->` $true_dest `, ` $false_dest"

    def __init__(
        self,
        input_op: SSAValue,
        count: int | IntegerAttr[I32],
        trueDest: Block,
        falseDest: Block,
        compareAtLeast: bool = False,
    ) -> None:
        if isinstance(count, int):
            count = IntegerAttr.from_int_and_width(count, 32)
        properties = dict[str, Attribute](count=count)
        if compareAtLeast:
            properties["compareAtLeast"] = UnitAttr()
        super().__init__(
            operands=[input_op],
            properties=properties,
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class IsNotNullOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpis_not_null-pdl_interpisnotnullop
    """

    name = "pdl_interp.is_not_null"
    traits = traits_def(IsTerminator())
    value = operand_def(AnyPDLTypeConstr)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = (
        "$value `:` type($value) attr-dict `->` $true_dest `, ` $false_dest"
    )

    def __init__(self, value: SSAValue, trueDest: Block, falseDest: Block) -> None:
        super().__init__(operands=[value], successors=[trueDest, falseDest])


@irdl_op_definition
class GetResultOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_result-pdl_interpgetresultop
    """

    name = "pdl_interp.get_result"
    index = prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType)

    assembly_format = "$index `of` $input_op attr-dict"

    def __init__(self, index: int | IntegerAttr[I32], input_op: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            operands=[input_op], properties={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class GetResultsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_results-pdl_interpgetresultsop
    """

    name = "pdl_interp.get_results"
    index = opt_prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType | RangeType[ValueType])

    # assembly_format = "($index^)? `of` $input_op `:` type($value) attr-dict"
    # TODO: Fix bug preventing this assebmly format from working: https://github.com/xdslproject/xdsl/issues/4136

    def __init__(
        self,
        index: int | IntegerAttr[I32] | None,
        input_op: SSAValue,
        result_type: ValueType | RangeType[ValueType],
    ) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            operands=[input_op],
            properties={"index": index},
            result_types=[result_type],
        )


@irdl_op_definition
class GetAttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_attribute-pdl_interpgetattributeop
    """

    name = "pdl_interp.get_attribute"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_op = operand_def(OperationType)
    value = result_def(AttributeType)

    assembly_format = "$name `of` $input_op attr-dict"

    def __init__(self, name: str | StringAttr, input_op: SSAValue) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(
            operands=[input_op],
            properties={"name": name},
            result_types=[AttributeType()],
        )


@irdl_op_definition
class CheckAttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_attribute-pdl_interpcheckattributeop
    """

    name = "pdl_interp.check_attribute"
    traits = traits_def(IsTerminator())
    constantValue = prop_def(Attribute)
    attribute = operand_def(AttributeType)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = (
        "$attribute `is` $constantValue attr-dict `->` $true_dest `, ` $false_dest"
    )

    def __init__(
        self,
        constantValue: Attribute,
        attribute: SSAValue,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        super().__init__(
            operands=[attribute],
            properties={"constantValue": constantValue},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class AreEqualOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpare_equal-pdl_interpareequalop
    """

    name = "pdl_interp.are_equal"
    traits = traits_def(IsTerminator())
    T: ClassVar = VarConstraint("T", AnyPDLTypeConstr)
    lhs = operand_def(T)
    rhs = operand_def(T)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = (
        "operands `:` type($lhs) attr-dict `->` $true_dest `, ` $false_dest"
    )

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, trueDest: Block, falseDest: Block
    ) -> None:
        super().__init__(operands=[lhs, rhs], successors=[trueDest, falseDest])


@irdl_op_definition
class RecordMatchOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interprecord_match-pdl_interprecordmatchop
    """

    name = "pdl_interp.record_match"
    traits = traits_def(IsTerminator())
    rewriter = prop_def(SymbolRefAttr)
    rootKind = opt_prop_def(StringAttr)
    generatedOps = opt_prop_def(ArrayAttr)
    benefit = prop_def(IntegerAttr[I16])

    inputs = var_operand_def(AnyPDLTypeConstr)
    matched_ops = var_operand_def(OperationType)

    dest = successor_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    assembly_format = (
        "$rewriter (`(` $inputs^ `:` type($inputs) `)`)? `:` `benefit` `(` $benefit `)` `,`"
        "(`generatedOps` `(` $generatedOps^ `)` `,`)? `loc` `(` `[` $matched_ops `]` `)`"
        "(`,` `root` `(` $rootKind^ `)`)? attr-dict `->` $dest"
    )

    def __init__(
        self,
        rewriter: str | SymbolRefAttr,
        root_kind: str | StringAttr,
        generated_ops: list[OperationType] | None,
        benefit: int | IntegerAttr[I16],
        inputs: Sequence[SSAValue],
        matched_ops: Sequence[SSAValue],
        dest: Block,
    ) -> None:
        if isinstance(rewriter, str):
            rewriter = SymbolRefAttr(rewriter)
        if isinstance(root_kind, str):
            root_kind = StringAttr(root_kind)
        if (
            generated_ops is None
        ):  # TODO: if generatedOps is actually optional (check this), we shouldn't even pass an empty list
            generated_ops = []
        if isinstance(benefit, int):
            benefit = IntegerAttr.from_int_and_width(benefit, 16)
        super().__init__(
            operands=[inputs, matched_ops],
            properties={
                "rewriter": rewriter,
                "rootKind": root_kind,
                "generatedOps": ArrayAttr(generated_ops),
                "benefit": benefit,
            },
            successors=[dest],
        )


@dataclass(frozen=True)
class ValueConstrFromResultConstr(
    GenericAttrConstraint[ValueType | RangeType[ValueType]]
):
    result_constr: GenericAttrConstraint[TypeType | RangeType[TypeType]]

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.result_constr.can_infer(var_constraint_names)

    def infer(self, context: ConstraintContext) -> ValueType | RangeType[ValueType]:
        result_type = self.result_constr.infer(context)
        if isinstance(result_type, RangeType):
            return RangeType(ValueType())
        return ValueType()

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if isa(attr, RangeType[ValueType]):
            result_type = RangeType(TypeType())
        elif isa(attr, ValueType):
            result_type = TypeType()
        else:
            raise VerifyException(
                f"Expected an attribute of type ValueType or RangeType[ValueType], but got {attr}"
            )
        return self.result_constr.verify(result_type, constraint_context)


@irdl_op_definition
class GetValueTypeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_value_type-pdl_interpgetvaluetypeop
    """

    name = "pdl_interp.get_value_type"
    T: ClassVar = VarConstraint("T", base(TypeType) | base(RangeType[TypeType]))
    value = operand_def(ValueConstrFromResultConstr(T))
    result = result_def(T)

    assembly_format = "`of` $value `:` type($result) attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(
            operands=[value],
            result_types=[
                RangeType(TypeType())
                if isinstance(value.type, RangeType)
                else TypeType()
            ],
        )


@irdl_op_definition
class ReplaceOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpreplace-pdl_interpreplaceop
    """

    name = "pdl_interp.replace"
    input_op = operand_def(OperationType)
    repl_values = var_operand_def(ValueType | RangeType[ValueType])

    assembly_format = (
        "$input_op `with` ` ` `(` ($repl_values^ `:` type($repl_values))? `)` attr-dict"
    )

    def __init__(self, input_op: SSAValue, repl_values: list[SSAValue]) -> None:
        super().__init__(operands=[input_op, repl_values])


@irdl_op_definition
class CreateAttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_attribute-pdl_interpcreateattributeop
    """

    name = "pdl_interp.create_attribute"
    value = prop_def(AnyAttr())
    attribute = result_def(AttributeType)

    assembly_format = "$value attr-dict-with-keyword"

    def __init__(self, value: Attribute) -> None:
        super().__init__(properties={"value": value}, result_types=[AttributeType()])


@irdl_op_definition
class CreateOperationOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_operation-pdl_interpcreateoperationop
    """

    name = "pdl_interp.create_operation"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_attribute_names = prop_def(ArrayAttr, prop_name="inputAttributeNames")
    inferred_result_types = opt_prop_def(UnitAttr, prop_name="inferredResultTypes")

    input_operands = var_operand_def(ValueType | RangeType[ValueType])
    input_attributes = var_operand_def(AttributeType | RangeType[AttributeType])
    input_result_types = var_operand_def(TypeType | RangeType[TypeType])

    result_op = result_def(OperationType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    # assembly_format = (
    #     "$name (`(` $input_operands^ `:` type($input_operands) `)`)?"
    #     "` `custom<CreateOperationOpAttributes>($inputAttributes, $inputAttributeNames)"
    #     "custom<CreateOperationOpResults>($inputResultTypes, type($inputResultTypes), $inferredResultTypes)"
    #     "attr-dict"
    # )
    # TODO: this assebly format is unsupported in xDSL because of the `custom` directives.

    def __init__(
        self,
        name: str | StringAttr,
        inferred_result_types: UnitAttr | None = None,
        input_attribute_names: Iterable[StringAttr] | None = None,
        input_operands: Sequence[SSAValue] | None = None,
        input_attributes: Sequence[SSAValue] | None = None,
        input_result_types: Sequence[SSAValue] | None = None,
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        if input_attribute_names is not None:
            input_attribute_names = ArrayAttr(input_attribute_names)
        if input_attribute_names is None:
            input_attribute_names = ArrayAttr([])

        if input_operands is None:
            input_operands = []
        if input_attributes is None:
            input_attributes = []
        if input_result_types is None:
            input_result_types = []

        super().__init__(
            operands=[input_operands, input_attributes, input_result_types],
            result_types=[OperationType()],
            properties={
                "name": name,
                "inferredResultTypes": inferred_result_types,
                "inputAttributeNames": input_attribute_names,
            }
            if inferred_result_types
            else {
                "name": name,
                "inputAttributeNames": input_attribute_names,
            },
        )


@irdl_op_definition
class GetDefiningOpOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_defining_op-pdl_interpgetdefiningopop
    """

    name = "pdl_interp.get_defining_op"
    value = operand_def(ValueType | ArrayAttr[ValueType])
    input_op = result_def(OperationType)

    assembly_format = "`of` $value `:` type($value) attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(operands=[value], result_types=[OperationType()])


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body

    @classmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfunc-pdl_interpfuncop
    """

    name = "pdl_interp.func"
    sym_name = prop_def(StringAttr)
    function_type = prop_def(FunctionType)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    body = region_def()

    traits = traits_def(
        IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()
    )

    def __init__(
        self,
        sym_name: str | StringAttr,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ) -> None:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))

        super().__init__(
            properties={
                "sym_name": sym_name,
                "function_type": function_type,
                "arg_attrs": arg_attrs,
                "res_attrs": res_attrs,
            },
            regions=[region],
        )


PDLInterp = Dialect(
    "pdl_interp",
    [
        GetOperandOp,
        FinalizeOp,
        CheckOperationNameOp,
        CheckOperandCountOp,
        CheckResultCountOp,
        IsNotNullOp,
        GetResultOp,
        GetResultsOp,
        GetAttributeOp,
        CheckAttributeOp,
        AreEqualOp,
        RecordMatchOp,
        GetValueTypeOp,
        ReplaceOp,
        CreateAttributeOp,
        CreateOperationOp,
        FuncOp,
        GetDefiningOpOp,
    ],
)
