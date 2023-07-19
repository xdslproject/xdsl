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
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    OptOperand,
    OptRegion,
    ParameterDef,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_region_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator, NoTerminator
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
        if operand.type != type:
            parser.raise_error(
                "Mismatched between operands and their types", pos, end_pos
            )

    return operands


def print_operands_with_types(printer: Printer, operands: Iterable[SSAValue]) -> None:
    printer.print_list(operands, printer.print)
    printer.print(" : ")
    printer.print_list([operand.type for operand in operands], printer.print)


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
    constraint_name: StringAttr = attr_def(StringAttr, attr_name="name")
    args: VarOperand = var_operand_def(AnyPDLType)

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

    name = "pdl.apply_native_rewrite"
    constraint_name: StringAttr = attr_def(StringAttr, attr_name="name")
    args: VarOperand = var_operand_def(AnyPDLType)
    res: VarOpResult = var_result_def(AnyPDLType)

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

    name = "pdl.attribute"
    value: Attribute | None = opt_attr_def(Attribute)
    value_type: OptOperand = opt_operand_def(TypeType)
    output: OpResult = result_def(AttributeType)

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

    name = "pdl.erase"
    op_value: Operand = operand_def(OperationType)

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

    name = "pdl.operand"
    value_type: OptOperand = opt_operand_def(TypeType)
    value: OpResult = result_def(ValueType)

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

    name = "pdl.operands"
    value_type: OptOperand = opt_operand_def(RangeType[TypeType])
    value: OpResult = result_def(RangeType[ValueType])

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

    name = "pdl.operation"
    opName: StringAttr | None = opt_attr_def(StringAttr)
    attributeValueNames: ArrayAttr[StringAttr] = attr_def(ArrayAttr[StringAttr])

    operand_values: VarOperand = var_operand_def(ValueType | RangeType[ValueType])
    attribute_values: VarOperand = var_operand_def(AttributeType)
    type_values: VarOperand = var_operand_def(TypeType | RangeType[TypeType])
    op: OpResult = result_def(OperationType)

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
        name = parser.parse_optional_str_literal()
        operands = []
        if parser.parse_optional_punctuation("(") is not None:
            operands = parse_operands_with_types(parser)
            parser.parse_punctuation(")")

        def parse_attribute_entry() -> tuple[str, SSAValue]:
            name = parser.parse_str_literal()
            parser.parse_punctuation("=")
            type = parser.parse_operand()
            return (name, type)

        attributes = parser.parse_optional_comma_separated_list(
            Parser.Delimiter.BRACES, parse_attribute_entry
        )
        if attributes is None:
            attributes = []
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

    name = "pdl.pattern"
    benefit: IntegerAttr[Annotated[IntegerType, IntegerType(16)]] = attr_def(
        IntegerAttr[Annotated[IntegerType, IntegerType(16)]]
    )
    sym_name: StringAttr | None = opt_attr_def(StringAttr)
    body: Region = region_def()

    def __init__(
        self,
        benefit: int | IntegerAttr[IntegerType],
        sym_name: str | StringAttr | None,
        body: Region | None = None,
    ):
        if isinstance(benefit, int):
            benefit = IntegerAttr(benefit, 16)
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if body is None:
            body = Region(Block())
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
        name = parser.parse_optional_symbol_name()
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

    name = "pdl.range"
    arguments: VarOperand = var_operand_def(AnyPDLType | RangeType[AnyPDLType])
    result: OpResult = result_def(RangeType[AnyPDLType])

    def verify_(self) -> None:
        def get_type_or_elem_type(arg: SSAValue) -> Attribute:
            if isa(arg.type, RangeType[AnyPDLType]):
                return arg.type.element_type
            else:
                return arg.type

        if len(self.arguments) > 0:
            elem_type = get_type_or_elem_type(self.result)

            for arg in self.arguments:
                if cur_elem_type := get_type_or_elem_type(arg) != elem_type:
                    raise VerifyException(
                        "All arguments must have the same type or be an array of the "
                        f"corresponding element type. First element type: {elem_type}"
                        f", current element type: {cur_elem_type}"
                    )

    def __init__(
        self,
        arguments: Sequence[SSAValue],
        result_type: Attribute | None = None,
    ) -> None:
        if result_type is None:
            if len(arguments) == 0:
                raise ValueError("Empty range constructions require a return type.")

            if isa(arguments[0].type, RangeType[AnyPDLType]):
                result_type = RangeType(arguments[0].type.element_type)
            elif isa(arguments[0].type, AnyPDLType):
                result_type = RangeType(arguments[0].type)
            else:
                raise ValueError(
                    f"Arguments of {self.name} are expected to be PDL types"
                )

        super().__init__(operands=[arguments], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> RangeOp:
        if parser.parse_optional_punctuation(":") is not None:
            return RangeOp([], parser.parse_attribute())

        arguments = parse_operands_with_types(parser)
        return RangeOp(arguments)

    def print(self, printer: Printer) -> None:
        if len(self.arguments) == 0:
            printer.print(" : ", self.result.type)
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

    name = "pdl.replace"
    op_value: Operand = operand_def(OperationType)
    repl_operation: OptOperand = opt_operand_def(OperationType)
    repl_values: VarOperand = var_operand_def(ValueType | ArrayAttr[ValueType])

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

    name = "pdl.result"
    index: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    parent_: Operand = operand_def(OperationType)
    val: OpResult = result_def(ValueType)

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

    name = "pdl.results"
    index: IntegerAttr[IntegerType] | None = opt_attr_def(IntegerAttr[IntegerType])
    parent_: Operand = operand_def(OperationType)
    val: OpResult = result_def(ValueType | RangeType[ValueType])

    def __init__(
        self,
        parent: SSAValue,
        index: int | IntegerAttr[IntegerType] | None = None,
        result_type: Attribute = RangeType(ValueType()),
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
        return ResultsOp(parent, index, result_type)

    def print(self, printer: Printer) -> None:
        if self.index is None:
            printer.print(" of ", self.parent_)
            return
        printer.print(
            " ", self.index.value.data, " of ", self.parent_, " -> ", self.val.type
        )


@irdl_op_definition
class RewriteOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrewrite-mlirpdlrewriteop
    """

    name = "pdl.rewrite"
    root: OptOperand = opt_operand_def(OperationType)
    # name of external rewriter function
    name_: StringAttr | None = opt_attr_def(StringAttr, attr_name="name")
    # parameters of external rewriter function
    external_args: VarOperand = var_operand_def(AnyPDLType)
    # body of inline rewriter function
    body: OptRegion = opt_region_def()

    irdl_options = [AttrSizedOperandSegments()]

    traits = frozenset([HasParent(PatternOp), NoTerminator(), IsTerminator()])

    def __init__(
        self,
        root: SSAValue | None,
        body: Region | type[Region.DEFAULT] | None = Region.DEFAULT,
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
        if body is Region.DEFAULT:
            regions.append(Region(Block()))
        elif isinstance(body, Region):
            regions.append(body)
        elif body is None:
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

        printer.print(" with ", self.name_)
        if len(self.external_args) != 0:
            printer.print("(")
            print_operands_with_types(printer, self.external_args)
            printer.print(")")


@irdl_op_definition
class TypeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-mlirpdltypeop
    """

    name = "pdl.type"
    constantType: Attribute | None = opt_attr_def(Attribute)
    result: OpResult = result_def(TypeType)

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

    name = "pdl.types"
    constantTypes: AnyArrayAttr | None = opt_attr_def(AnyArrayAttr)
    result: OpResult = result_def(RangeType[TypeType])

    def __init__(self, constant_types: Iterable[Attribute] | None = None) -> None:
        if constant_types is not None:
            attributes = {"constantTypes": ArrayAttr(constant_types)}
        else:
            attributes = {}
        super().__init__(
            attributes=attributes,
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
