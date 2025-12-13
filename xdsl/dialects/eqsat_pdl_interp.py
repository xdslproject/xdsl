"""
A dialect that extends pdl_interp with eqsat-specific operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

from xdsl.dialects.builtin import (
    I16,
    I32,
    ArrayAttr,
    IntegerAttr,
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
from xdsl.ir import Attribute, Block, Dialect, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_successor_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


@irdl_op_definition
class GetResultOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_result-pdl_interpgetresultop).
    """

    name = "eqsat_pdl_interp.get_result"
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

    name = "eqsat_pdl_interp.get_results"
    index = opt_prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType | RangeType[ValueType])

    assembly_format = "($index^)? `of` $input_op `:` type($value) attr-dict"

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
class GetDefiningOpOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_defining_op-pdl_interpgetdefiningopop).
    """

    name = "eqsat_pdl_interp.get_defining_op"
    value = operand_def(ValueType | RangeType[ValueType])
    input_op = result_def(OperationType)

    assembly_format = "`of` $value `:` type($value) attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(operands=[value], result_types=[OperationType()])


@irdl_op_definition
class ReplaceOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpreplace-pdl_interpreplaceop).
    """

    name = "eqsat_pdl_interp.replace"
    input_op = operand_def(OperationType)
    repl_values = var_operand_def(ValueType | RangeType[ValueType])

    assembly_format = (
        "$input_op `with` ` ` `(` ($repl_values^ `:` type($repl_values))? `)` attr-dict"
    )

    def __init__(self, input_op: SSAValue, repl_values: list[SSAValue]) -> None:
        super().__init__(operands=[input_op, repl_values])


@irdl_op_definition
class CreateOperationOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_operation-pdl_interpcreateoperationop).
    """

    name = "eqsat_pdl_interp.create_operation"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_attribute_names = prop_def(
        ArrayAttr[StringAttr], prop_name="inputAttributeNames"
    )
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


@irdl_op_definition
class RecordMatchOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interprecord_match-pdl_interprecordmatchop).
    """

    name = "eqsat_pdl_interp.record_match"
    traits = traits_def(IsTerminator())
    rewriter = prop_def(SymbolRefAttr)
    rootKind = opt_prop_def(StringAttr)
    generatedOps = opt_prop_def(ArrayAttr[StringAttr])
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


@irdl_op_definition
class FinalizeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfinalize-pdl_interpfinalizeop).
    """

    name = "eqsat_pdl_interp.finalize"
    traits = traits_def(IsTerminator())

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ChooseOp(IRDLOperation):
    """
    This operation can be used in pdl_interp matchers and
    integrates with the backtracking mechanism. It holds multiple
    "choices" (successors). When this operation is encountered,
    a BacktrackPoint is stored, and the choice is visited.
    When this execution of this choice eventually finalizes, the
    backtracking logic will jump to the next choice, until all
    choices are exhausted. Finally, the default successor is visited.
    """

    name = "eqsat_pdl_interp.choose"
    default_dest = successor_def()
    choices = var_successor_def()
    traits = traits_def(IsTerminator())
    assembly_format = "`from` $choices `then` $default_dest attr-dict"

    def __init__(self, choices: Sequence[Block], default: Block):
        super().__init__(
            successors=[default, choices],
        )


EqSatPDLInterp = Dialect(
    "eqsat_pdl_interp",
    [
        GetResultOp,
        GetResultsOp,
        GetDefiningOpOp,
        ReplaceOp,
        CreateOperationOp,
        RecordMatchOp,
        FinalizeOp,
        ChooseOp,
    ],
)
