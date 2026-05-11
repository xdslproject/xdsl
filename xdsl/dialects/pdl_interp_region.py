from __future__ import annotations

from typing import Iterable, Sequence, cast

from xdsl.dialects.builtin import (
    I32,
    IntegerAttr,
    StringAttr, ArrayAttr, UnitAttr,
)
from xdsl.dialects.irdl import AttributeType
from xdsl.dialects.pdl import (
    OperationType,
    ValueType, RangeType, TypeType, AnyPDLTypeConstr, AnyPDLType,
)
from xdsl.dialects.pdl_region import (
    RegionType,
)
from xdsl.ir import (
    Dialect,
    SSAValue, Attribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def, opt_prop_def, AttrSizedOperandSegments, opt_operand_def, base
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_op_definition
class GetRegionOp(IRDLOperation):
    name = "pdl_interp_region.get_region"
    input_op = operand_def(OperationType)
    index = prop_def(IntegerAttr[I32])
    value = result_def(RegionType)

    assembly_format = "$index `of` $input_op `:` type($value) attr-dict"

    def __init__(self, index: int | IntegerAttr[I32], input_op: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            properties={"index": index},
            operands=[input_op],
            result_types=[RegionType],
        )


@irdl_op_definition
class InlineRegionOp(IRDLOperation):
    name = "pdl_interp_region.inline_region"
    input_op = operand_def(OperationType)
    repl_values = var_operand_def(RegionType)
    value = result_def(ValueType)

    assembly_format = (
        "$input_op `with` ` ` `(` ($repl_values^ `:` type($repl_values))? `)` attr-dict"
    )

    def __init__(self, input_op: SSAValue, repl_values: SSAValue[RegionType]) -> None:
        super().__init__(operands=[input_op, repl_values])

@irdl_op_definition
class GetOperationOp(IRDLOperation):
    name = "pdl_interp_region.get_operation"
    index = prop_def(IntegerAttr[I32])
    opt_name = prop_def(StringAttr, prop_name="opt_name")
    input_region = operand_def(RegionType)
    opt_operands = var_operand_def(AnyPDLType | RangeType[AnyPDLType])

    result_op = result_def(OperationType)

    assembly_format = (
        "`(` ($opt_operands^ `:` type($opt_operands))? `)` "
        "` called` $opt_name ` `"
        "$index `of` $input_region "
        "attr-dict"
    )
    " (`called` $opt_name^)?"

    def __init__(
        self,
        index: int | IntegerAttr[I32],
        input_region: SSAValue,
        opt_operands: Sequence[SSAValue] | None = None,
        opt_attributes: SSAValue | None = None,
        opt_name: str | StringAttr | None = None,
        opt_type: SSAValue | None = None,
    ):
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        if isinstance(opt_name, str):
            opt_name = StringAttr(opt_name)

        super().__init__(
            properties={"index": index, "name": opt_name},
            operands=[input_region, opt_operands, opt_attributes, opt_type],
            result_types=[OperationType()],
        )


@irdl_op_definition
class CreateRegionOp(IRDLOperation):
    name = "pdl_interp_region.create_region"
    opt_operands = var_operand_def(OperationType | RangeType[OperationType])

    result_op = result_def(RegionType)

    assembly_format = (
        "`(` ($opt_operands^ `:` type($opt_operands))? `)` "
        "attr-dict"
    )

    def __init__(
        self,
        opt_operands: Sequence[SSAValue] | None = None,
    ):
        super().__init__(
            operands=[opt_operands],
            result_types=[RegionType()]
        )


@irdl_op_definition
class InsertOpIntoRegionOp(IRDLOperation):
    name = "pdl_interp_region.insert_op_into_region"
    opt_operand = operand_def(OperationType)
    opt_before = opt_operand_def(OperationType)
    region = operand_def(RegionType)

    result_op = result_def(RegionType)

    assembly_format = (
        "`(` $opt_operand `:` type($opt_operand) `)` "
        "(`before` $opt_before^)? "
        "`of` $region "
        "attr-dict"
    )

    def __init__(
        self,
        opt_operand: SSAValue,
        region: SSAValue,
        opt_before: SSAValue | None = None,
    ):
        super().__init__(
            operands=[opt_operand, opt_before, region],
            result_types=[RegionType()]
        )


@irdl_op_definition
class DeleteOpFromRegionOp(IRDLOperation):
    name = "pdl_interp_region.delete_op_from_region"
    opt_operand = operand_def(OperationType)
    region = operand_def(RegionType)

    result_op = result_def(RegionType)

    assembly_format = (
        "`(` $opt_operand `:` type($opt_operand) `)` "
        "`of` $region "
        "attr-dict"
    )

    def __init__(
        self,
        opt_operand: SSAValue,
        region: SSAValue,
    ):
        super().__init__(
            operands=[opt_operand, region],
            result_types=[RegionType()]
        )


@irdl_op_definition
class CreateOperationRegionOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_operation-pdl_interpcreateoperationop).
    """

    name = "pdl_interp_region.create_operation_with_region"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_attribute_names = prop_def(ArrayAttr, prop_name="inputAttributeNames")
    inferred_result_types = opt_prop_def(UnitAttr, prop_name="inferredResultTypes")

    input_operands = var_operand_def(ValueType | RangeType[ValueType] | RegionType)
    input_attributes = var_operand_def(AttributeType | RangeType[AttributeType])
    input_result_types = var_operand_def(TypeType | RangeType[TypeType])

    result_op = result_def(OperationType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

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
    def parse(cls, parser: Parser) -> CreateOperationRegionOp:
        name = parser.parse_attribute()

        input_operands = CreateOperationRegionOp._parse_input_list(parser)

        input_attribute_names = None
        input_attributes = None
        attributes = parser.parse_optional_comma_separated_list(
            delimiter=Parser.Delimiter.BRACES,
            parse=lambda: CreateOperationRegionOp._parse_attr(parser),
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
                input_result_types = CreateOperationRegionOp._parse_input_list(parser)

        op = CreateOperationRegionOp.build(
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
            CreateOperationRegionOp._print_input_list(printer, self.input_operands)
        if self.input_attributes:
            printer.print_string(" {", indent=0)
            printer.print_list(
                zip(
                    cast(tuple[StringAttr], self.input_attribute_names.data),
                    self.input_attributes,
                ),
                lambda value: CreateOperationRegionOp._print_attr(printer, value),
            )
            printer.print_string("}", indent=0)
        if self.inferred_result_types:
            assert not self.input_result_types
            printer.print_string(" -> <inferred>", indent=0)
        elif self.input_result_types:
            printer.print_string(" -> ", indent=0)
            CreateOperationRegionOp._print_input_list(printer, self.input_result_types)


PDLInterpRegion = Dialect(
    "pdl_interp_region",
    [
        GetRegionOp,
        InlineRegionOp,
        GetOperationOp,
        CreateOperationRegionOp,
        CreateRegionOp,
        InsertOpIntoRegionOp,
        DeleteOpFromRegionOp,
    ],
)
