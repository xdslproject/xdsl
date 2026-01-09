from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I16,
    I32,
    ArrayAttr,
    BoolAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator, NoTerminator, OptionalSymbolOpInterface
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
    printer.print_list(operands, printer.print_ssa_value)
    printer.print_string(" : ")
    printer.print_list(operands, lambda o: printer.print_attribute(o.type))


def has_binding_use(op: Operation) -> bool:
    """
    Returns true if the given operation is used by a "binding" pdl operation.
    """
    for result in op.results:
        for use in result.uses:
            if not isinstance(use.operation, ResultOp | ResultsOp) or has_binding_use(
                use.operation
            ):
                return True
    return False


def verify_has_binding_use(op: Operation) -> None:
    """
    Raise an exception if the operation is in the main matcher body and is
    not used by a "binding" pdl operation.
    """
    if not isinstance(op.parent_op(), PatternOp):
        return
    if not has_binding_use(op):
        raise VerifyException(
            "expected a bindable user when defined in the matcher body of a "
            "`pdl.pattern`"
        )


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
AnyPDLTypeConstr = (
    base(AttributeType) | base(OperationType) | base(TypeType) | base(ValueType)
)

_RangeT = TypeVar(
    "_RangeT",
    bound=AttributeType | OperationType | TypeType | ValueType,
    covariant=True,
)


@irdl_attr_definition
class RangeType(ParametrizedAttribute, TypeAttribute, Generic[_RangeT]):
    name = "pdl.range"
    element_type: _RangeT

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        parser.parse_punctuation("<")
        if parser.parse_optional_keyword("attribute") is not None:
            element_type = AttributeType()
        elif parser.parse_optional_keyword("operation") is not None:
            element_type = OperationType()
        elif parser.parse_optional_keyword("type") is not None:
            element_type = TypeType()
        elif parser.parse_optional_keyword("value") is not None:
            element_type = ValueType()
        else:
            parser.raise_error("expected PDL element type for range")
        parser.parse_punctuation(">")
        return [element_type]

    def print_parameters(self, printer: Printer) -> None:
        match self.element_type:
            case AttributeType():
                printer.print_string("<attribute>")
            case OperationType():
                printer.print_string("<operation>")
            case TypeType():
                printer.print_string("<type>")
            case ValueType():
                printer.print_string("<value>")


@irdl_op_definition
class ApplyNativeConstraintOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_constraint-pdlapplynativeconstraintop).
    """

    name = "pdl.apply_native_constraint"
    constraint_name = prop_def(StringAttr, prop_name="name")
    is_negated = prop_def(
        BoolAttr, prop_name="isNegated", default_value=BoolAttr.from_bool(False)
    )
    args = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    res = var_result_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = "$name `(` $args `:` type($args) `)` (`:` type($res)^)? attr-dict"

    def __init__(
        self,
        name: str | StringAttr,
        args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        is_negated: bool = False,
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(
            result_types=[result_types],
            operands=[args],
            properties={"name": name, "isNegated": BoolAttr.from_bool(is_negated)},
        )


@irdl_op_definition
class ApplyNativeRewriteOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlapply_native_rewrite-pdlapplynativerewriteop).
    """

    name = "pdl.apply_native_rewrite"
    constraint_name = prop_def(StringAttr, prop_name="name")
    args = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    res = var_result_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))

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
            properties={"name": name},
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
        printer.print_string(" ")
        printer.print_string_literal(self.constraint_name.data)
        with printer.in_parens():
            print_operands_with_types(printer, self.operands)
        if len(self.results) != 0:
            printer.print_string(" : ")
            printer.print_list(self.result_types, printer.print_attribute)


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-pdlattributeop).
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-pdlattributeop
    """

    name = "pdl.attribute"
    value = opt_prop_def()
    value_type = opt_operand_def(TypeType)
    output = result_def(AttributeType)

    assembly_format = "(`:` $value_type^)? (`=` $value^)? attr-dict-with-keyword"

    def verify_(self):
        if self.value is not None and self.value_type is not None:
            raise VerifyException(
                f"{self.name} cannot both specify an expected attribute "
                "via a constant value and an expected type."
            )
        if self.value is None and isinstance(self.parent_op(), RewriteOp):
            raise VerifyException(
                "expected constant value when specified within a `pdl.rewrite`"
            )
        verify_has_binding_use(self)

    def __init__(self, value: Attribute | SSAValue | None = None) -> None:
        """
        The given value is either the expected attribute, if given an attribute, or the
        expected attribute type, if given an SSAValue.
        """
        properties: dict[str, Attribute] = {}
        operands: list[SSAValue | None] = [None]
        if isinstance(value, Attribute):
            properties["value"] = value
        elif isinstance(value, SSAValue):
            operands = [value]

        super().__init__(
            operands=operands, properties=properties, result_types=[AttributeType()]
        )


@irdl_op_definition
class EraseOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-pdleraseop).
    """

    name = "pdl.erase"
    op_value = operand_def(OperationType)

    assembly_format = "$op_value attr-dict"

    def __init__(self, op_value: SSAValue) -> None:
        super().__init__(operands=[op_value])


@irdl_op_definition
class OperandOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-pdloperandop).
    """

    name = "pdl.operand"
    value_type = opt_operand_def(TypeType)
    value = result_def(ValueType)

    assembly_format = "(`:` $value_type^)? attr-dict"

    def __init__(self, value_type: SSAValue | None = None) -> None:
        super().__init__(operands=[value_type], result_types=[ValueType()])

    def verify_(self):
        verify_has_binding_use(self)


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-pdloperandsop).
    """

    name = "pdl.operands"
    value_type = opt_operand_def(RangeType[TypeType])
    value = result_def(RangeType[ValueType])

    assembly_format = "(`:` $value_type^)? attr-dict"

    def __init__(self, value_type: SSAValue | None) -> None:
        super().__init__(operands=[value_type], result_types=[RangeType(ValueType())])

    def verify_(self):
        verify_has_binding_use(self)


@irdl_op_definition
class OperationOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperation-pdloperationop).
    """

    name = "pdl.operation"
    opName = opt_prop_def(StringAttr)
    attributeValueNames = prop_def(ArrayAttr[StringAttr])

    operand_values = var_operand_def(base(ValueType) | base(RangeType[ValueType]))
    attribute_values = var_operand_def(AttributeType)
    type_values = var_operand_def(base(TypeType) | base(RangeType[TypeType]))
    op = result_def(OperationType)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

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
            properties={
                "attributeValueNames": attribute_value_names,
                "opName": op_name,
            },
        )

    def verify_(self):
        is_within_rewrite: bool = isinstance(self.parent_op(), RewriteOp)
        if is_within_rewrite and self.opName is None:
            raise VerifyException(
                "must have an operation name when nested within a `pdl.rewrite`"
            )
        if len(self.attributeValueNames) != len(self.attribute_values):
            raise VerifyException(
                "expected the same number of attribute values and attribute "
                f"names, got {len(self.attributeValueNames)} names and "
                f"{len(self.attribute_values)} values"
            )
        verify_has_binding_use(self)

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
            printer.print_string(" ")
            printer.print_attribute(self.opName)

        if len(self.operand_values) != 0:
            printer.print_string(" ")
            with printer.in_parens():
                print_operands_with_types(printer, self.operand_values)

        def print_attribute_entry(entry: tuple[StringAttr, SSAValue]):
            printer.print_attribute(entry[0])
            printer.print_string(" = ")
            printer.print_ssa_value(entry[1])

        if len(self.attributeValueNames) != 0:
            printer.print_string(" ")
            with printer.in_braces():
                printer.print_list(
                    zip(self.attributeValueNames, self.attribute_values),
                    print_attribute_entry,
                )

        if len(self.type_values) != 0:
            printer.print_string(" -> ")
            with printer.in_parens():
                print_operands_with_types(printer, self.type_values)


def _visit_pdl_ops(op: Operation, visited: set[Operation]):
    """
    Visit all pdl.operands, pdl.results, and pdl.operations connected to the
    given operation in a pdl.pattern.
    """
    # We only look at operations within a `pdl.pattern`.
    if not isinstance(op.parent_op(), PatternOp):
        return

    if op in visited:
        return

    visited.add(op)

    # Traverse the operands
    if isinstance(op, OperationOp):
        for value in op.operand_values:
            assert isinstance(owner := value.owner, Operation)
            _visit_pdl_ops(owner, visited)
    elif isinstance(op, ResultOp | ResultsOp):
        assert isinstance(owner := op.parent_.owner, Operation)
        _visit_pdl_ops(owner, visited)

    # Traverse the users
    for result in op.results:
        for user in result.uses:
            _visit_pdl_ops(user.operation, visited)


def _has_user_in_rewrite(op: Operation) -> bool:
    """Check if an operation has a user in a pdl.rewrite"""
    for result in op.results:
        for use in result.uses:
            if isinstance(use.operation.parent_op(), RewriteOp):
                return True
    return False


@irdl_op_definition
class PatternOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlpattern-pdlpatternop).
    """

    name = "pdl.pattern"
    benefit = prop_def(IntegerAttr[I16])
    sym_name = opt_prop_def(StringAttr)
    body = region_def("single_block")

    traits = traits_def(OptionalSymbolOpInterface())

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
            properties={
                "benefit": benefit,
                "sym_name": sym_name,
            },
            regions=[body],
            result_types=[],
        )

    def verify_(self):
        # Check for the correct terminator.
        if not isinstance(self.body.block.last_op, RewriteOp):
            raise VerifyException("expected body to terminate with a `pdl.rewrite`")

        # Check that there is at least one `pdl.operation`.
        if not any(isinstance(op, OperationOp) for op in self.body.block.ops):
            raise VerifyException(
                "the pattern must contain at least one `pdl.operation`"
            )

        # Get the connected component by traversing the graph in the first
        # PDL operation, operand, or result used in a `pdl.rewrite`. The other
        # operations will be detected via other means with a better error handling
        # (expected bindable user).
        first = True
        visited: set[Operation] = set()
        for op in self.body.block.ops:
            if not isinstance(
                op, OperandOp | OperandsOp | ResultOp | ResultsOp | OperationOp
            ):
                continue
            if not _has_user_in_rewrite(op):
                continue

            if first:
                first = False
                _visit_pdl_ops(op, visited)
            if op not in visited:
                raise VerifyException(
                    "Operations in a `pdl.pattern` must form a connected component"
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
            printer.print_string(" @")
            printer.print_string(self.sym_name.data)
        printer.print_string(f" : benefit({self.benefit.value.data}) ")
        printer.print_region(self.body)


@irdl_op_definition
class RangeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrange-pdlrangeop).
    """

    name = "pdl.range"
    arguments = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    result = result_def(RangeType[AnyPDLType])

    traits = lazy_traits_def(lambda: (HasParent(RewriteOp),))

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
            elif AnyPDLTypeConstr.verifies(arguments[0].type):
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
            printer.print_string(" : ")
            printer.print_attribute(self.result.type)
            return
        printer.print_string(" ")
        print_operands_with_types(printer, self.arguments)


@irdl_op_definition
class ReplaceOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlreplace-pdlreplaceop).

    `pdl.replace` operations are used within `pdl.rewrite` regions to specify
    that an input operation should be marked as replaced. The semantics of this
    operation correspond with the `replaceOp` method on a `PatternRewriter`. The
    set of replacement values can be either:
    * a single `Operation` (`replOperation` should be populated)
      - The operation will be replaced with the results of this operation.
    * a set of `Value`s (`replValues` should be populated)
      - The operation will be replaced with these values.
    """

    name = "pdl.replace"
    op_value = operand_def(OperationType)
    repl_operation = opt_operand_def(OperationType)
    repl_values = var_operand_def(base(ValueType) | base(ArrayAttr[ValueType]))

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    assembly_format = (
        "$op_value `with` ` ` "
        "(`(` $repl_values^ `:` type($repl_values) `)`)? $repl_operation attr-dict"
    )

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
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresult-pdlresultop).
    """

    name = "pdl.result"
    index = prop_def(IntegerAttr[I32])
    parent_ = operand_def(OperationType)
    val = result_def(ValueType)

    assembly_format = "$index `of` $parent_ attr-dict"

    def __init__(self, index: int | IntegerAttr[IntegerType], parent: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, 32)
        super().__init__(
            operands=[parent], properties={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresults-pdlresultsop).
    """

    name = "pdl.results"
    index = opt_prop_def(IntegerAttr[I32])
    parent_ = operand_def(OperationType)
    val = result_def(base(ValueType) | base(RangeType[ValueType]))

    def __init__(
        self,
        parent: SSAValue,
        index: int | IntegerAttr[IntegerType] | None = None,
        result_type: Attribute = RangeType(ValueType()),
    ) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, 32)
        super().__init__(
            operands=[parent], result_types=[result_type], properties={"index": index}
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
            printer.print_string(" of ")
            printer.print_ssa_value(self.parent_)
        else:
            printer.print_string(f" {self.index.value.data} of ")
            printer.print_ssa_value(self.parent_)
            printer.print_string(" -> ")
            printer.print_attribute(self.val.type)


@irdl_op_definition
class RewriteOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlrewrite-pdlrewriteop).
    """

    name = "pdl.rewrite"
    root = opt_operand_def(OperationType)
    # name of external rewriter function
    name_ = opt_prop_def(StringAttr, prop_name="name")
    # parameters of external rewriter function
    external_args = var_operand_def(AnyPDLTypeConstr)
    # body of inline rewriter function
    body = region_def()

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = traits_def(HasParent(PatternOp), NoTerminator(), IsTerminator())

    assembly_format = (
        "($root^)? "
        "(`with` $name^ (`(` $external_args^ `:` type($external_args) `)`)?)?"
        "($body^)? attr-dict-with-keyword"
    )

    def __init__(
        self,
        root: SSAValue | None,
        body: Region | type[Region.DEFAULT] = Region.DEFAULT,
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

        properties: dict[str, Attribute] = {}
        if name is not None:
            properties["name"] = name

        super().__init__(
            result_types=[],
            operands=operands,
            properties=properties,
            regions=regions,
        )


@irdl_op_definition
class TypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltype-pdltypeop).
    """

    name = "pdl.type"
    constantType = opt_prop_def(TypeAttribute)
    result = result_def(TypeType)

    assembly_format = "attr-dict (`:` $constantType^)?"

    def __init__(self, constant_type: Attribute | None = None) -> None:
        super().__init__(
            properties={"constantType": constant_type}, result_types=[TypeType()]
        )

    def verify_(self):
        verify_has_binding_use(self)


@irdl_op_definition
class TypesOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/#pdltypes-pdltypesop).
    """

    name = "pdl.types"
    constantTypes = opt_prop_def(ArrayAttr[TypeAttribute])
    result = result_def(RangeType[TypeType])

    assembly_format = "attr-dict (`:` $constantTypes^)?"

    def __init__(self, constant_types: Iterable[Attribute] | None = None) -> None:
        if constant_types is not None:
            properties = {"constantTypes": ArrayAttr(constant_types)}
        else:
            properties = {}
        super().__init__(
            properties=properties,
            result_types=[RangeType(TypeType())],
        )

    def verify_(self):
        verify_has_binding_use(self)


PDL = Dialect(
    "pdl",
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
