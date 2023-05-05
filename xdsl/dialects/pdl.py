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
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


def parse_operands_with_types(parser: Parser) -> list[SSAValue]:
    """
    Parse a list of operands with types of the following format:
    `operand1, operand2 : type1, type2`
    At least one operand is expected.
    """
    pos = parser.pos
    operands = parser.parse_comma_separated_list(
        Parser.Delimiter.NONE, parser.parse_operand
    )
    parser.parse_punctuation(":")
    types = parser.parse_comma_separated_list(
        Parser.Delimiter.NONE, parser.parse_attribute
    )
    end_pos = parser.pos
    if len(operands) != len(types):
        parser.raise_error(
            "Mismatched between the numbers of operands and types", pos, end_pos
        )
    for operand, type in zip(operands, types):
        if operand.typ != type:
            raise Exception("Mismatched between operands and their types", pos, end_pos)

    return operands


def print_operands_with_types(printer: Printer, operands: Iterable[SSAValue]) -> None:
    printer.print_list(operands, printer.print)
    printer.print(" : ")
    printer.print_list([operand.typ for operand in operands], printer.print)


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

    name: str = "pdl.apply_native_constraint"
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

    @classmethod
    def parse(cls, parser: Parser) -> ApplyNativeConstraintOp:
        name = parser.parse_str_literal()
        parser.parse_punctuation("(")
        operands = parse_operands_with_types(parser)
        parser.parse_punctuation(")")
        return ApplyNativeConstraintOp(name, operands)

    def print(self, printer: Printer) -> None:
        printer.print_string_literal(self.constraint_name.data)
        printer.print("(")
        print_operands_with_types(printer, self.operands)
        printer.print(")")


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

    @classmethod
    def parse(cls, parser: Parser) -> ApplyNativeRewriteOp:
        name = parser.parse_str_literal()
        parser.parse_punctuation("(")
        operands = parse_operands_with_types(parser)
        parser.parse_punctuation(")")
        result_types = []
        if parser.parse_optional_punctuation(":") is not None:
            result_types = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parser.parse_attribute
            )
        return ApplyNativeRewriteOp(name, operands, result_types)

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_string_literal(self.constraint_name.data)
        printer.print("(")
        print_operands_with_types(printer, self.operands)
        printer.print(")")
        if len(self.results) != 0:
            printer.print(" : ")
            printer.print_list(self.results, printer.print)


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
    """

    name: str = "pdl.attribute"
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

    @classmethod
    def parse(cls, parser: Parser) -> AttributeOp:
        value: Operand | Attribute | None = None
        if parser.parse_optional_punctuation(":"):
            value = parser.parse_operand()
        elif parser.parse_optional_punctuation("="):
            value = parser.parse_attribute()

        return AttributeOp(value)

    def print(self, printer: Printer) -> None:
        if self.value is not None:
            printer.print(" = ", self.value)
        elif self.value_type is not None:
            printer.print(" : ", self.value_type)


@irdl_op_definition
class EraseOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-mlirpdleraseop
    """

    name: str = "pdl.erase"
    op_value: Annotated[Operand, OperationType]

    def __init__(self, op_value: SSAValue) -> None:
        super().__init__(operands=[op_value])

    @classmethod
    def parse(cls, parser: Parser) -> EraseOp:
        op_value = parser.parse_operand()
        return EraseOp(op_value)

    def print(self, printer: Printer) -> None:
        printer.print(" ", self.op_value)


@irdl_op_definition
class OperandOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-mlirpdloperandop
    """

    name: str = "pdl.operand"
    value_type: Annotated[OptOperand, TypeType]
    value: Annotated[OpResult, ValueType]

    def __init__(self, value_type: SSAValue | None = None) -> None:
        super().__init__(operands=[value_type], result_types=[ValueType()])

    @classmethod
    def parse(cls, parser: Parser) -> OperandOp:
        value = None
        if parser.parse_optional_punctuation(":") is not None:
            value = parser.parse_operand()

        return OperandOp(value)

    def print(self, printer: Printer) -> None:
        if self.value_type is not None:
            printer.print(" : ", self.value_type)


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-mlirpdloperandsop
    """

    name: str = "pdl.operands"
    value_type: Annotated[OptOperand, RangeType[TypeType]]
    value: Annotated[OpResult, RangeType[ValueType]]

    def __init__(self, value_type: SSAValue | None) -> None:
        super().__init__(operands=[value_type], result_types=[RangeType(ValueType())])

    @classmethod
    def parse(cls, parser: Parser) -> OperandsOp:
        value_type = None
        if parser.parse_optional_punctuation(":") is not None:
            value_type = parser.parse_operand()

        return OperandsOp(value_type)

    def print(self, printer: Printer) -> None:
        if self.value_type is not None:
            printer.print(" : ", self.value_type)


@irdl_op_definition
class OperationOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperation-mlirpdloperationop
    """

    name: str = "pdl.operation"
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
        attribute_value_names: Iterable[StringAttr] | None = None,
        operand_values: Sequence[SSAValue] | None = None,
        attribute_values: Sequence[SSAValue] | None = None,
        type_values: Sequence[SSAValue] | None = None,
    ):
        if isinstance(op_name, str):
            op_name = StringAttr(op_name)
        if attribute_value_names is not None:
            attribute_value_names = ArrayAttr(attribute_value_names)
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

    @classmethod
    def parse(cls, parser: Parser) -> OperationOp:
        name = parser.try_parse_string_literal()
        if name is not None:
            name = name.string_contents
        operands = []
        if parser.parse_optional_punctuation("(") is not None:
            operands = parse_operands_with_types(parser)
            parser.parse_punctuation(")")

        def parse_pattribute_entry() -> tuple[str, SSAValue]:
            name = parser.parse_str_literal()
            parser.parse_punctuation("=")
            type = parser.parse_operand()
            return (name, type)

        attributes = []
        if parser.parse_optional_punctuation("{"):
            attributes = parser.parse_comma_separated_list(
                Parser.Delimiter.NONE, parse_pattribute_entry
            )
            parser.parse_punctuation("}")
        attribute_names = [StringAttr(attr[0]) for attr in attributes]
        attribute_values = [attr[1] for attr in attributes]

        results = []
        if parser.parse_optional_punctuation("->"):
            parser.parse_punctuation("(")
            results = parse_operands_with_types(parser)
            parser.parse_punctuation(")")

        return OperationOp(name, attribute_names, operands, attribute_values, results)

    def print(self, printer: Printer) -> None:
        if self.opName is not None:
            printer.print(" ", self.opName)

        if len(self.operand_values) != 0:
            printer.print(" (")
            print_operands_with_types(printer, self.operand_values)
            printer.print(")")

        def print_attribute_entry(entry: tuple[StringAttr, SSAValue]):
            printer.print(entry[0], " = ", entry[1])

        if len(self.attributeValueNames) != 0:
            printer.print(" {")
            printer.print_list(
                zip(self.attributeValueNames, self.attribute_values),
                print_attribute_entry,
            )
            printer.print("}")

        if len(self.type_values) != 0:
            printer.print(" -> (")
            print_operands_with_types(printer, self.type_values)
            printer.print(")")


@irdl_op_definition
class PatternOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlpattern-mlirpdlpatternop
    """

    name: str = "pdl.pattern"
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

    @classmethod
    def parse(cls, parser: Parser) -> PatternOp:
        name = parser.try_parse_single_reference()
        if name is not None:
            name = name.text
        parser.parse_punctuation(":")
        parser.parse_keyword("benefit")
        parser.parse_punctuation("(")
        benefit = parser.parse_integer()
        parser.parse_punctuation(")")
        body = parser.parse_region()
        return PatternOp(benefit, name, body)

    def print(self, printer: Printer) -> None:
        if self.sym_name is not None:
            printer.print(" @", self.sym_name.data)
        printer.print(" : benefit(", self.benefit.value.data, ") ", self.body)


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
        result_type: Attribute | None = None,
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

    @classmethod
    def parse(cls, parser: Parser) -> RangeOp:
        if parser.parse_optional_punctuation(":") is not None:
            return RangeOp([], parser.parse_attribute())

        arguments = parse_operands_with_types(parser)
        return RangeOp(arguments)

    def print(self, printer: Printer) -> None:
        if len(self.arguments) == 0:
            printer.print(" : ", self.result.typ)
            return

        print_operands_with_types(printer, self.arguments)


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

    @classmethod
    def parse(cls, parser: Parser) -> ReplaceOp:
        root = parser.parse_operand()
        parser.parse_keyword("with")
        if (repl_op := parser.parse_optional_operand()) is not None:
            return ReplaceOp(root, repl_op)

        parser.parse_punctuation("(")
        repl_values = parse_operands_with_types(parser)
        parser.parse_punctuation(")")
        return ReplaceOp(root, repl_values=repl_values)

    def printer(self, printer: Printer) -> None:
        printer.print(self.op_value, " with ")
        if self.repl_operation is not None:
            printer.print(self.repl_operation)
            return
        printer.print("(")
        print_operands_with_types(printer, self.repl_values)
        printer.print(")")


@irdl_op_definition
class ResultOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresult-mlirpdlresultop
    """

    name: str = "pdl.result"
    index: OpAttr[IntegerAttr[Annotated[IntegerType, i32]]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType]

    def __init__(self, index: int | IntegerAttr[IntegerType], parent: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, 32)
        super().__init__(
            operands=[parent], attributes={"index": index}, result_types=[ValueType()]
        )

    @classmethod
    def parse(cls, parser: Parser) -> ResultOp:
        index = parser.parse_integer()
        parser.parse_keyword("of")
        parent = parser.parse_operand()
        return ResultOp(index, parent)

    def print(self, printer: Printer) -> None:
        printer.print(" ", self.index.value.data, " of ", self.parent_)


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresults-mlirpdlresultsop
    """

    name: str = "pdl.results"
    index: OptOpAttr[IntegerAttr[IntegerType]]
    parent_: Annotated[Operand, OperationType]
    val: Annotated[OpResult, ValueType | RangeType[ValueType]]

    def __init__(
        self,
        parent: SSAValue,
        result_type: Attribute = RangeType(ValueType()),
        index: int | IntegerAttr[IntegerType] | None = None,
    ) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, 32)
        super().__init__(
            operands=[parent], result_types=[result_type], attributes={"index": index}
        )

    @classmethod
    def parse(cls, parser: Parser) -> ResultsOp:
        if parser.parse_optional_keyword("of") is not None:
            parent = parser.parse_operand()
            return ResultsOp(parent)
        index = parser.parse_integer()
        parser.parse_keyword("of")
        parent = parser.parse_operand()
        parser.parse_punctuation("->")
        result_type = parser.parse_attribute()
        return ResultsOp(parent, result_type, index)

    def print(self, printer: Printer) -> None:
        if self.index is None:
            printer.print(" of ", self.parent_)
            return
        printer.print(
            " ", self.index.value.data, " of ", self.parent_, " -> ", self.val.typ
        )


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
        root: SSAValue | None,
        body: Region | Block.BlockCallback | None = None,
        name: str | StringAttr | None = None,
        external_args: Sequence[SSAValue] = (),
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

    @classmethod
    def parse(cls, parser: Parser) -> RewriteOp:
        root = parser.parse_optional_operand()

        if parser.parse_optional_keyword("with") is None:
            body = parser.parse_region()
            return RewriteOp(root, body)

        name = parser.parse_str_literal()
        external_args = []
        if parser.parse_optional_punctuation("(") is not None:
            external_args = parse_operands_with_types(parser)
            parser.parse_punctuation(")")

        return RewriteOp(root, None, name, external_args)

    def print(self, printer: Printer) -> None:
        if self.root is not None:
            printer.print(" ", self.root)

        if self.body is not None:
            printer.print(" ", self.body)
            return

        printer.print(" with ", self.name)
        if len(self.external_args) != 0:
            printer.print("(")
            print_operands_with_types(printer, self.external_args)
            printer.print(")")


@irdl_op_definition
class TypeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-mlirpdltypeop
    """

    name: str = "pdl.type"
    constantType: OptOpAttr[Attribute]
    result: Annotated[OpResult, TypeType]

    def __init__(self, constant_type: Attribute | None = None) -> None:
        super().__init__(
            attributes={"constantType": constant_type}, result_types=[TypeType()]
        )

    @classmethod
    def parse(cls, parser: Parser) -> TypeOp:
        if parser.parse_optional_punctuation(":") is None:
            return TypeOp()
        constant_type = parser.parse_attribute()
        return TypeOp(constant_type)

    def print(self, printer: Printer) -> None:
        if self.constantType is not None:
            printer.print(" : ", self.constantType)


@irdl_op_definition
class TypesOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltypes-mlirpdltypesop
    """

    name: str = "pdl.types"
    constantTypes: OptOpAttr[AnyArrayAttr]
    result: Annotated[OpResult, RangeType[TypeType]]

    def __init__(self, constant_types: Iterable[Attribute] = ()) -> None:
        super().__init__(
            attributes={"constantTypes": ArrayAttr(constant_types)},
            result_types=[RangeType(TypeType())],
        )

    @classmethod
    def parse(cls, parser: Parser) -> TypesOp:
        if parser.parse_optional_punctuation(":") is None:
            return TypesOp()
        begin_attr_pos = parser.pos
        constant_types = parser.parse_attribute()
        if not isa(constant_types, AnyArrayAttr):
            parser.raise_error("Array attribute expected", begin_attr_pos, parser.pos)
        return TypesOp(constant_types)

    def print(self, printer: Printer) -> None:
        if self.constantTypes is not None:
            printer.print(" : ", self.constantTypes)


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
