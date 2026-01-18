from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import ClassVar, cast

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I16,
    I32,
    ArrayAttr,
    BoolAttr,
    ContainerOf,
    DenseIntElementsAttr,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    VectorType,
)
from xdsl.dialects.pdl import (
    AnyPDLType,
    AnyPDLTypeConstr,
    AttributeType,
    OperationType,
    RangeType,
    TypeType,
    ValueType,
)
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    AttrConstraint,
    AttrSizedOperandSegments,
    ConstraintContext,
    IntConstraint,
    IRDLOperation,
    ParsePropInAttrDict,
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
    var_result_def,
    var_successor_def,
)
from xdsl.irdl.declarative_assembly_format import (
    CustomDirective,
    ParsingState,
    PrintingState,
    TypeDirective,
    VariadicOperandVariable,
    irdl_custom_directive,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_operand-pdl_interpgetoperandop).
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
class GetOperandsOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_operands-pdl_interpgetoperandsop).
    """

    name = "pdl_interp.get_operands"
    index = opt_prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType | RangeType[ValueType])

    # TODO: assembly format doesn't work due to a bug:
    # https://github.com/xdslproject/xdsl/issues/5562
    # assembly_format = "($index^)? `of` $input_op `:` type($value) attr-dict"

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

    @classmethod
    def parse(cls, parser: Parser) -> GetOperandsOp:
        index = parser.parse_optional_integer()
        if index is not None:
            index = IntegerAttr.from_int_and_width(index, 32)
        parser.parse_characters("of")
        input_op = parser.parse_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        return GetOperandsOp.build(
            operands=(input_op,),
            properties={"index": index},
            result_types=(result_type,),
        )

    def print(self, printer: Printer):
        if self.index is not None:
            printer.print_string(" ", indent=0)
            self.index.print_without_type(printer)
        printer.print_string(" of ", indent=0)
        printer.print_operand(self.input_op)
        printer.print_string(" : ", indent=0)
        printer.print_attribute(self.value.type)


@irdl_op_definition
class FinalizeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfinalize-pdl_interpfinalizeop).
    """

    name = "pdl_interp.finalize"
    traits = traits_def(IsTerminator())

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class CheckOperationNameOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_operation_name-pdl_interpcheckoperationnameop).
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_operand_count-pdl_interpcheckoperandcountop).
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_result_count-pdl_interpcheckresultcountop).
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpis_not_null-pdl_interpisnotnullop).
    """

    name = "pdl_interp.is_not_null"
    traits = traits_def(IsTerminator())
    value = operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_result-pdl_interpgetresultop).
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_results-pdl_interpgetresultsop).
    """

    name = "pdl_interp.get_results"
    index = opt_prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType | RangeType[ValueType])

    # assembly_format = "($index^)? `of` $input_op `:` type($value) attr-dict"
    # TODO: Fix bug preventing this assebmly format from working: https://github.com/xdslproject/xdsl/issues/4136.

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

    @classmethod
    def parse(cls, parser: Parser) -> GetResultsOp:
        index = parser.parse_optional_integer()
        if index is not None:
            index = IntegerAttr.from_int_and_width(index, 32)
        parser.parse_characters("of")
        input_op = parser.parse_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        return GetResultsOp.build(
            operands=(input_op,),
            properties={"index": index},
            result_types=(result_type,),
        )

    def print(self, printer: Printer):
        if self.index is not None:
            printer.print_string(" ", indent=0)
            self.index.print_without_type(printer)
        printer.print_string(" of ", indent=0)
        printer.print_operand(self.input_op)
        printer.print_string(" : ", indent=0)
        printer.print_attribute(self.value.type)


@irdl_op_definition
class GetAttributeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_attribute-pdl_interpgetattributeop).
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
class GetAttributeTypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_attribute_type-pdl_interpgetattributetypeop).
    """

    name = "pdl_interp.get_attribute_type"
    value = operand_def(AttributeType)
    result = result_def(TypeType)

    assembly_format = "`of` $value attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(
            operands=[value],
            result_types=[TypeType()],
        )


@irdl_op_definition
class CheckAttributeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_attribute-pdl_interpcheckattributeop).
    """

    name = "pdl_interp.check_attribute"
    traits = traits_def(IsTerminator())
    constantValue = prop_def()
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
class CheckTypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_type-pdl_interpchecktypeop).
    """

    name = "pdl_interp.check_type"
    traits = traits_def(IsTerminator())
    type = prop_def(TypeAttribute)
    value = operand_def(TypeType)
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = "$value `is` $type attr-dict `->` $true_dest `, ` $false_dest"

    def __init__(
        self, type: TypeAttribute, value: SSAValue, trueDest: Block, falseDest: Block
    ) -> None:
        super().__init__(
            operands=[value],
            properties={"type": type},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class CheckTypesOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_types-pdl_interpchecktypesop).
    """

    name = "pdl_interp.check_types"
    traits = traits_def(IsTerminator())
    types = prop_def(ArrayAttr[TypeAttribute])
    value = operand_def(RangeType[TypeType])
    true_dest = successor_def()
    false_dest = successor_def()

    assembly_format = "$value `are` $types attr-dict `->` $true_dest `, ` $false_dest"

    def __init__(
        self,
        types: ArrayAttr[TypeAttribute] | Sequence[TypeAttribute],
        value: SSAValue,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        if not isinstance(types, ArrayAttr):
            types = ArrayAttr(types)
        super().__init__(
            operands=[value],
            properties={"types": types},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class AreEqualOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpare_equal-pdl_interpareequalop).
    """

    name = "pdl_interp.are_equal"
    traits = traits_def(IsTerminator())
    T: ClassVar = VarConstraint("T", AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
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
class ApplyConstraintOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpapply_constraint-pdl_interpapplyconstraintop).
    """

    name = "pdl_interp.apply_constraint"
    traits = traits_def(IsTerminator())
    constraint_name = prop_def(StringAttr, prop_name="name")
    is_negated = prop_def(
        BoolAttr, prop_name="isNegated", default_value=BoolAttr.from_bool(False)
    )
    args = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    results_ = var_result_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    true_dest = successor_def()
    false_dest = successor_def()
    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = "$name `(` $args `:` type($args) `)` (`:` type($results_)^)? attr-dict `->` $true_dest `,` $false_dest"

    def __init__(
        self,
        constraint_name: str | StringAttr,
        args: Sequence[SSAValue],
        true_dest: Block,
        false_dest: Block,
        res_types: Sequence[AnyPDLType | RangeType[AnyPDLType]] = (),
        is_negated: bool | BoolAttr = False,
    ) -> None:
        if isinstance(constraint_name, str):
            constraint_name = StringAttr(constraint_name)
        if isinstance(is_negated, bool):
            is_negated = BoolAttr.from_bool(is_negated)
        super().__init__(
            operands=[args],
            properties={
                "name": constraint_name,
                "isNegated": is_negated,
            },
            result_types=[
                res_types,
            ],
            successors=[true_dest, false_dest],
        )


@irdl_op_definition
class ApplyRewriteOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpapply_rewrite-pdl_interpapplyrewriteop).
    """

    name = "pdl_interp.apply_rewrite"
    rewrite_name = prop_def(StringAttr, prop_name="name")
    args = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    results_ = var_result_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "$name (`(` $args^ `:` type($args) `)`)? (`:` type($results_)^)? attr-dict"
    )

    def __init__(
        self,
        rewrite_name: str | StringAttr,
        args: Sequence[SSAValue],
        res_types: Sequence[AnyPDLType | RangeType[AnyPDLType]] = (),
    ) -> None:
        if isinstance(rewrite_name, str):
            rewrite_name = StringAttr(rewrite_name)
        super().__init__(
            operands=[args],
            properties={
                "name": rewrite_name,
            },
            result_types=[
                res_types,
            ],
        )


@irdl_op_definition
class RecordMatchOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interprecord_match-pdl_interprecordmatchop).
    """

    name = "pdl_interp.record_match"
    traits = traits_def(IsTerminator())
    rewriter = prop_def(SymbolRefAttr)
    rootKind = opt_prop_def(StringAttr)
    generatedOps = opt_prop_def(ArrayAttr[StringAttr])
    benefit = prop_def(IntegerAttr[I16])

    inputs = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    matched_ops = var_operand_def(OperationType)

    dest = successor_def()

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    assembly_format = (
        "$rewriter (`(` $inputs^ `:` type($inputs) `)`)? `:` `benefit` `(` $benefit `)` `,`"
        "(`generatedOps` `(` $generatedOps^ `)` `,`)? `loc` `(` `[` $matched_ops `]` `)`"
        "(`,` `root` `(` $rootKind^ `)`)? attr-dict `->` $dest"
    )

    def __init__(
        self,
        rewriter: str | SymbolRefAttr,
        root_kind: str | StringAttr | None,
        generated_ops: ArrayAttr[StringAttr] | None,
        benefit: int | IntegerAttr[I16],
        inputs: Sequence[SSAValue],
        matched_ops: Sequence[SSAValue],
        dest: Block,
    ) -> None:
        if isinstance(rewriter, str):
            rewriter = SymbolRefAttr(rewriter)
        if isinstance(root_kind, str):
            root_kind = StringAttr(root_kind)
        if isinstance(benefit, int):
            benefit = IntegerAttr.from_int_and_width(benefit, 16)
        super().__init__(
            operands=[inputs, matched_ops],
            properties={
                "rewriter": rewriter,
                "rootKind": root_kind,
                "generatedOps": generated_ops,
                "benefit": benefit,
            },
            successors=[dest],
        )


@dataclass(frozen=True)
class ValueConstrFromResultConstr(AttrConstraint[ValueType | RangeType[ValueType]]):
    result_constr: AttrConstraint[TypeType | RangeType[TypeType]]

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
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

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[ValueType | RangeType[ValueType]]:
        return ValueConstrFromResultConstr(
            self.result_constr.mapping_type_vars(type_var_mapping)
        )


@irdl_op_definition
class GetValueTypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_value_type-pdl_interpgetvaluetypeop).
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpreplace-pdl_interpreplaceop).
    """

    name = "pdl_interp.replace"
    input_op = operand_def(OperationType)
    repl_values = var_operand_def(ValueType | RangeType[ValueType])

    assembly_format = (
        "$input_op `with` ` ` `(` ($repl_values^ `:` type($repl_values))? `)` attr-dict"
    )

    def __init__(self, input_op: SSAValue, repl_values: Sequence[SSAValue]) -> None:
        super().__init__(operands=[input_op, repl_values])


@irdl_op_definition
class EraseOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interperase-pdl_interpreaseop).
    """

    name = "pdl_interp.erase"
    input_op = operand_def(OperationType)

    assembly_format = "$input_op attr-dict"

    def __init__(self, input_op: SSAValue) -> None:
        super().__init__(operands=[input_op])


@irdl_op_definition
class CreateAttributeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_attribute-pdl_interpcreateattributeop).
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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_operation-pdl_interpcreateoperationop).
    """

    name = "pdl_interp.create_operation"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_attribute_names = prop_def(
        ArrayAttr[StringAttr], prop_name="inputAttributeNames"
    )
    inferred_result_types = opt_prop_def(UnitAttr, prop_name="inferredResultTypes")

    input_operands = var_operand_def(ValueType | RangeType[ValueType])
    input_attributes = var_operand_def(AttributeType | RangeType[AttributeType])
    input_result_types = var_operand_def(TypeType | RangeType[TypeType])

    result_op = result_def(OperationType)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

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

    @staticmethod
    def _parse_attr(parser: Parser) -> tuple[Attribute, SSAValue]:
        attrname = parser.parse_attribute()
        parser.parse_punctuation("=")
        operand = parser.parse_operand()
        return (attrname, operand)

    @staticmethod
    def _parse_input_list(parser: Parser) -> list[SSAValue]:
        values: list[SSAValue] = []
        if parser.parse_optional_punctuation("("):
            values = parser.parse_comma_separated_list(
                delimiter=Parser.Delimiter.NONE,
                parse=lambda: parser.parse_operand(),
            )
            parser.parse_punctuation(":")
            parser.parse_comma_separated_list(
                delimiter=Parser.Delimiter.NONE,
                parse=lambda: parser.parse_type(),
            )
            parser.parse_punctuation(")")
        return values

    @classmethod
    def parse(cls, parser: Parser) -> CreateOperationOp:
        name = parser.parse_attribute()

        input_operands = CreateOperationOp._parse_input_list(parser)

        input_attribute_names = None
        input_attributes = None
        attributes = parser.parse_optional_comma_separated_list(
            delimiter=Parser.Delimiter.BRACES,
            parse=lambda: CreateOperationOp._parse_attr(parser),
        )
        if attributes is not None:
            input_attribute_names = [i[0] for i in attributes]
            input_attributes = [i[1] for i in attributes]
        else:
            input_attribute_names = []
            input_attributes = []
        input_attribute_names = ArrayAttr(input_attribute_names)

        input_result_types = None
        inferred_result_types = None
        if parser.parse_optional_punctuation("->") is not None:
            if parser.parse_optional_punctuation("<"):
                parser.parse_characters("inferred")
                parser.parse_punctuation(">")
                inferred_result_types = UnitAttr()
            else:
                input_result_types = CreateOperationOp._parse_input_list(parser)

        op = CreateOperationOp.build(
            operands=(input_operands, input_attributes, input_result_types),
            properties={
                "name": name,
                "inputAttributeNames": input_attribute_names,
            }
            if inferred_result_types is None
            else {
                "name": name,
                "inferredResultTypes": inferred_result_types,
                "inputAttributeNames": input_attribute_names,
            },
            result_types=(OperationType(),),
        )
        return op

    @staticmethod
    def _print_input_list(printer: Printer, values: Iterable[SSAValue]):
        printer.print_string("(", indent=0)
        printer.print_list(values, printer.print_operand)
        printer.print_string(" : ", indent=0)
        printer.print_list(values, lambda op: printer.print_attribute(op.type))
        printer.print_string(")", indent=0)

    @staticmethod
    def _print_attr(printer: Printer, value: tuple[StringAttr, SSAValue]):
        printer.print_attribute(value[0])
        printer.print_string(" = ", indent=0)
        printer.print_operand(value[1])

    def print(self, printer: Printer):
        printer.print_string(" ", indent=0)
        printer.print_attribute(self.constraint_name)
        if self.input_operands:
            CreateOperationOp._print_input_list(printer, self.input_operands)
        if self.input_attributes:
            printer.print_string(" {", indent=0)
            printer.print_list(
                zip(
                    cast(tuple[StringAttr], self.input_attribute_names.data),
                    self.input_attributes,
                ),
                lambda value: CreateOperationOp._print_attr(printer, value),
            )
            printer.print_string("}", indent=0)
        if self.inferred_result_types:
            assert not self.input_result_types
            printer.print_string(" -> <inferred>", indent=0)
        elif self.input_result_types:
            printer.print_string(" -> ", indent=0)
            CreateOperationOp._print_input_list(printer, self.input_result_types)


@irdl_custom_directive
class RangeTypeDirective(CustomDirective):
    """
    Custom directive for parsing/printing range types in CreateRangeOp.
    """

    arguments_type: TypeDirective
    result_type: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        args_inner = self.arguments_type.inner
        assert isinstance(args_inner, VariadicOperandVariable)

        arg_types = state.operand_types[args_inner.index]

        if arg_types:
            # Infer result type from first argument (getRangeElementTypeOrSelf)
            first_type = arg_types[0]
            if isa(first_type, RangeType[AnyPDLType]):
                element_type = first_type.element_type
            else:
                assert isinstance(first_type, AnyPDLType)
                element_type = first_type
            result_type = RangeType(element_type)
            self.result_type.set(state, (result_type,))
            return False  # No input consumed during inference
        else:
            # Parse `: type` for result when no arguments
            parser.parse_punctuation(":")
            result_type = parser.parse_type()
            self.result_type.set(state, (result_type,))
            return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        arg_types = self.arguments_type.get(op)
        if not arg_types:
            # Print `: result_type` only when no arguments
            printer.print_string(" : ")
            result_types = self.result_type.get(op)
            printer.print_attribute(result_types[0])


@irdl_op_definition
class CreateRangeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_range-pdl_interpcreaterangeop).
    """

    name = "pdl_interp.create_range"
    arguments = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    result = result_def(RangeType[AnyPDLType])

    custom_directives = (RangeTypeDirective,)

    assembly_format = (
        "($arguments^ `:` type($arguments))?"
        "custom<RangeTypeDirective>(ref(type($arguments)), type($result))"
        "attr-dict"
    )

    def __init__(
        self,
        arguments: Sequence[SSAValue],
        result_type: RangeType[AnyPDLType],
    ) -> None:
        super().__init__(
            operands=[arguments],
            result_types=[result_type],
        )


@irdl_op_definition
class GetDefiningOpOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_defining_op-pdl_interpgetdefiningopop).
    """

    name = "pdl_interp.get_defining_op"
    value = operand_def(ValueType | RangeType[ValueType])
    input_op = result_def(OperationType)

    assembly_format = "`of` $value `:` type($value) attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(operands=[value], result_types=[OperationType()])


@irdl_op_definition
class SwitchOperationNameOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpswitch_operation_name-pdl_interpswitchoperationnameop).
    """

    name = "pdl_interp.switch_operation_name"

    case_values = prop_def(ArrayAttr[StringAttr], prop_name="caseValues")

    input_op = operand_def(OperationType)

    default_dest = successor_def()
    cases = var_successor_def()

    traits = traits_def(IsTerminator())
    assembly_format = (
        "`of` $input_op `to` $caseValues `(` $cases `)` attr-dict `->` $default_dest"
    )

    def __init__(
        self,
        case_values: ArrayAttr[StringAttr] | Iterable[StringAttr],
        input_op: SSAValue,
        default_dest: Block,
        cases: Sequence[Block],
    ) -> None:
        case_values = ArrayAttr(case_values)
        super().__init__(
            operands=[input_op],
            properties={"caseValues": case_values},
            successors=[default_dest, cases],
        )


@irdl_op_definition
class SwitchOperandCountOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpswitch_operand_count-pdl_interpswitchoperandcountop).
    """

    name = "pdl_interp.switch_operand_count"

    case_values = prop_def(DenseIntElementsAttr, prop_name="caseValues")

    input_op = operand_def(OperationType)

    default_dest = successor_def()
    cases = var_successor_def()

    traits = traits_def(IsTerminator())
    assembly_format = (
        "`of` $input_op `to` $caseValues `(` $cases `)` attr-dict `->` $default_dest"
    )

    def __init__(
        self,
        case_values: DenseIntElementsAttr | Sequence[int],
        input_op: SSAValue,
        default_dest: Block,
        cases: Sequence[Block],
    ) -> None:
        if not isa(case_values, DenseIntElementsAttr):
            assert isinstance(case_values, Sequence)
            case_values = DenseIntElementsAttr.from_list(
                VectorType(IntegerType(32), (len(case_values),)), case_values
            )

        super().__init__(
            operands=[input_op],
            properties={"caseValues": case_values},
            successors=[default_dest, cases],
        )


@irdl_op_definition
class SwitchResultCountOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpswitch_result_count-pdl_interpswitchresultcountop).
    """

    name = "pdl_interp.switch_result_count"

    case_values = prop_def(DenseIntElementsAttr, prop_name="caseValues")

    input_op = operand_def(OperationType)

    default_dest = successor_def()
    cases = var_successor_def()

    traits = traits_def(IsTerminator())
    assembly_format = (
        "`of` $input_op `to` $caseValues `(` $cases `)` attr-dict `->` $default_dest"
    )

    def __init__(
        self,
        case_values: DenseIntElementsAttr | Sequence[int],
        input_op: SSAValue,
        default_dest: Block,
        cases: Sequence[Block],
    ) -> None:
        if not isa(case_values, DenseIntElementsAttr):
            assert isinstance(case_values, Sequence)
            case_values = DenseIntElementsAttr.from_list(
                VectorType(IntegerType(32), (len(case_values),)), case_values
            )
        super().__init__(
            operands=[input_op],
            properties={"caseValues": case_values},
            successors=[default_dest, cases],
        )


@irdl_op_definition
class SwitchTypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpswitch_type-pdl_interpswitchtypeop).
    """

    name = "pdl_interp.switch_type"

    case_values = prop_def(ArrayAttr[TypeAttribute], prop_name="caseValues")

    value = operand_def(TypeType)

    default_dest = successor_def()
    cases = var_successor_def()

    traits = traits_def(IsTerminator())
    assembly_format = (
        "$value `to` $caseValues `(` $cases `)` attr-dict `->` $default_dest"
    )

    def __init__(
        self,
        case_values: ArrayAttr[TypeAttribute] | Sequence[TypeAttribute],
        value: SSAValue,
        default_dest: Block,
        cases: Sequence[Block],
    ) -> None:
        if not isinstance(case_values, ArrayAttr):
            case_values = ArrayAttr(case_values)
        super().__init__(
            operands=[value],
            properties={"caseValues": case_values},
            successors=[default_dest, cases],
        )


@irdl_op_definition
class SwitchTypesOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpswitch_types-pdl_interpswitchtypesop).
    """

    name = "pdl_interp.switch_types"

    case_values = prop_def(ArrayAttr[ArrayAttr[TypeAttribute]], prop_name="caseValues")

    value = operand_def(RangeType[TypeType])

    default_dest = successor_def()
    cases = var_successor_def()

    traits = traits_def(IsTerminator())
    assembly_format = (
        "$value `to` $caseValues `(` $cases `)` attr-dict `->` $default_dest"
    )

    def __init__(
        self,
        case_values: ArrayAttr[ArrayAttr[TypeAttribute]]
        | Sequence[ArrayAttr[TypeAttribute] | Sequence[TypeAttribute]],
        value: SSAValue,
        default_dest: Block,
        cases: Sequence[Block],
    ) -> None:
        if not isinstance(case_values, ArrayAttr):
            case_values = ArrayAttr(
                [v if isinstance(v, ArrayAttr) else ArrayAttr(v) for v in case_values]
            )
        super().__init__(
            operands=[value],
            properties={"caseValues": case_values},
            successors=[default_dest, cases],
        )


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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfunc-pdl_interpfuncop).
    """

    name = "pdl_interp.func"
    sym_name = prop_def(SymbolNameConstraint())
    function_type = prop_def(FunctionType)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    body = region_def()

    traits = traits_def(
        IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()
    )

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
            res_attrs,
        ) = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type")
        )
        func = FuncOp(
            sym_name=name,
            function_type=(input_types, return_types),
            region=region,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "arg_attrs",
            ),
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


@irdl_op_definition
class SwitchAttributeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpswitch_attribute-pdl_interpswitchattributeop).
    """

    name = "pdl_interp.switch_attribute"

    attribute = operand_def(AttributeType)
    caseValues = prop_def(ArrayAttr)
    defaultDest = successor_def()
    cases = var_successor_def()

    traits = traits_def(IsTerminator())

    assembly_format = (
        "$attribute `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest"
    )

    def __init__(
        self,
        attribute: SSAValue,
        case_values: ArrayAttr,
        default_dest: Block,
        cases: list[Block],
    ) -> None:
        super().__init__(
            operands=[attribute],
            properties={"caseValues": case_values},
            successors=[default_dest, cases],
        )


@irdl_op_definition
class CreateTypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_type-pdl_interpcreatetypeop).
    """

    name = "pdl_interp.create_type"

    value = prop_def(TypeAttribute)
    result = result_def(TypeType)

    assembly_format = "$value attr-dict"

    def __init__(self, value: TypeAttribute) -> None:
        super().__init__(
            properties={"value": value},
            result_types=[TypeType()],
        )


@irdl_op_definition
class CreateTypesOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_types-pdl_interpcreatetypesop).
    """

    name = "pdl_interp.create_types"

    value = prop_def(ArrayAttr)
    result = result_def(RangeType[TypeType])

    assembly_format = "$value attr-dict"

    def __init__(self, value: ArrayAttr) -> None:
        super().__init__(
            properties={"value": value},
            result_types=[RangeType(TypeType())],
        )


@irdl_op_definition
class ContinueOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcontinue-pdl_interpcontinueop).
    """

    name = "pdl_interp.continue"
    traits = traits_def(IsTerminator())

    assembly_format = "attr-dict"

    def __init__(self) -> None:
        super().__init__()


@irdl_op_definition
class ForEachOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpforeach-pdl_interpforeachop).
    """

    name = "pdl_interp.foreach"
    traits = traits_def(IsTerminator())

    values = operand_def(RangeType[AnyPDLType])
    region = region_def()
    successor = successor_def()

    def __init__(self, values: SSAValue, successor: Block, region: Region) -> None:
        super().__init__(operands=[values], successors=[successor], regions=[region])

    @classmethod
    def parse(cls, parser: Parser) -> ForEachOp:
        arg = parser.parse_argument()
        parser.parse_characters("in")
        values = parser.parse_operand()
        region = parser.parse_region(arguments=[arg])
        parser.parse_punctuation("->")
        successor = parser.parse_successor()
        attrs = parser.parse_optional_attr_dict()

        op = ForEachOp(values, successor, region)
        if attrs:
            op.attributes = attrs
        return op

    def print(self, printer: Printer):
        loop_var = self.region.blocks[0].args[0]
        printer.print_string(" ")
        printer.print_ssa_value(loop_var)
        printer.print_string(" : ")
        printer.print_attribute(loop_var.type)
        printer.print_string(" in ")
        printer.print_operand(self.values)
        printer.print_string(" ")
        printer.print_region(self.region, print_entry_block_args=False)
        printer.print_string(" -> ")
        printer.print_block_name(self.successor)
        printer.print_op_attributes(self.attributes)

    def verify_(self) -> None:
        if not self.region.blocks:
            raise VerifyException("Region must not be empty")

        block = self.region.blocks[0]
        if len(block.args) != 1:
            raise VerifyException("Region must have exactly one argument")

        arg_type = block.args[0].type

        assert isa(self.values, SSAValue[RangeType[AnyPDLType]])
        if self.values.type.element_type != arg_type:
            raise VerifyException(
                f"Region argument type {arg_type} does not match "
                f"range element type {self.values.type.element_type}"
            )


PDLInterp = Dialect(
    "pdl_interp",
    [
        GetOperandOp,
        GetOperandsOp,
        FinalizeOp,
        CheckOperationNameOp,
        CheckOperandCountOp,
        CheckResultCountOp,
        CheckTypeOp,
        CheckTypesOp,
        IsNotNullOp,
        GetResultOp,
        GetResultsOp,
        GetAttributeOp,
        GetAttributeTypeOp,
        CheckAttributeOp,
        AreEqualOp,
        ApplyConstraintOp,
        ApplyRewriteOp,
        RecordMatchOp,
        GetValueTypeOp,
        ReplaceOp,
        EraseOp,
        CreateAttributeOp,
        CreateOperationOp,
        CreateRangeOp,
        SwitchOperationNameOp,
        SwitchOperandCountOp,
        SwitchResultCountOp,
        SwitchTypeOp,
        SwitchTypesOp,
        SwitchAttributeOp,
        CreateTypeOp,
        CreateTypesOp,
        FuncOp,
        GetDefiningOpOp,
        ForEachOp,
        ContinueOp,
    ],
)
