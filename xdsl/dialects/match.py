"""
Imperative matching with implicit failure and deferred rewrite candidates.

Based on Jules Merckx's Match dialect at llvm-project revision
e9b3a5db2e3b574d4e6ff69d213eb2305cbe3002:
https://github.com/jumerckx/llvm-project/tree/e9b3a5db2e3b574d4e6ff69d213eb2305cbe3002/mlir/include/mlir/Dialect/Match/IR

This is an incremental port. Unlike MLIR, xDSL does not elide the dialect prefix
of operations inside a matcher.
"""

from collections.abc import Sequence
from typing import ClassVar, Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I32,
    IntegerAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    i16,
    i32,
)
from xdsl.dialects.pdl import (
    AnyPDLType,
    AnyPDLTypeConstr,
    AttributeType,
    OperationType,
    RangeType,
    ValueType,
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
    AtLeast,
    IRDLOperation,
    ParamAttrConstraint,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasAncestor,
    IsolatedFromAbove,
    NoTerminator,
    OptionalSymbolOpInterface,
    Pure,
    SymbolTable,
    SymbolUserOpInterface,
)
from xdsl.utils.exceptions import VerifyException

_OptionalT = TypeVar(
    "_OptionalT",
    bound=AnyPDLType | RangeType[AnyPDLType],
    default=AnyPDLType | RangeType[AnyPDLType],
    covariant=True,
)


@irdl_attr_definition
class OptionalType(ParametrizedAttribute, TypeAttribute, Generic[_OptionalT]):
    """A possibly-null PDL handle, unwrapped by `match.is_not_null`."""

    name = "match.optional"
    inner_type: _OptionalT


@irdl_op_definition
class MatcherOp(IRDLOperation):
    """An ordered matcher with one root operation and at least one success."""

    name = "match.matcher"

    sym_name = opt_prop_def(SymbolNameConstraint())
    sym_visibility = opt_prop_def(StringAttr)
    body = region_def("single_block")

    traits = traits_def(
        IsolatedFromAbove(), NoTerminator(), OptionalSymbolOpInterface()
    )

    def __init__(
        self,
        body: Region | None = None,
        sym_name: str | StringAttr | None = None,
        sym_visibility: str | StringAttr | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(sym_visibility, str):
            sym_visibility = StringAttr(sym_visibility)
        if body is None:
            body = Region(Block(arg_types=[OperationType()]))
        super().__init__(
            properties={"sym_name": sym_name, "sym_visibility": sym_visibility},
            regions=[body],
        )

    def verify_(self) -> None:
        if len(self.body.block.args) != 1:
            raise VerifyException("expected body block to have exactly one argument")
        if self.body.block.args[0].type != OperationType():
            raise VerifyException(
                "expected body block argument to be of type !pdl.operation"
            )
        if not any(isinstance(op, SuccessOp) for op in self.body.walk()):
            raise VerifyException(
                "expected matcher to contain at least one `match.success` "
                "(directly or transitively)"
            )

    @classmethod
    def parse(cls, parser: Parser) -> "MatcherOp":
        visibility = parser.parse_optional_str_literal()
        name = parser.parse_optional_symbol_name()
        parser.parse_keyword("root")
        parser.parse_punctuation("(")
        root = parser.parse_argument()
        parser.parse_punctuation(")")
        body = parser.parse_region([root])
        attributes = parser.parse_optional_attr_dict_with_keyword(
            ("sym_name", "sym_visibility")
        )
        op = cls(body, name, visibility)
        if attributes is not None:
            op.attributes.update(attributes.data)
        return op

    def print(self, printer: Printer) -> None:
        if self.sym_visibility is not None:
            printer.print_string(" ")
            printer.print_attribute(self.sym_visibility)
        if self.sym_name is not None:
            printer.print_string(" ")
            printer.print_symbol_name(self.sym_name.data)
        printer.print_string(" root(")
        if (block := self.body.blocks.first) is not None and block.args:
            printer.print_block_argument(block.args[0])
        printer.print_string(") ")
        printer.print_region(
            self.body, print_entry_block_args=False, print_empty_block=False
        )
        printer.print_op_attributes(self.attributes, print_keyword=True)


class SuccessOpSymbolUserOpInterface(SymbolUserOpInterface):
    def verify(self, op: Operation) -> None:
        assert isinstance(op, SuccessOp)
        if SymbolTable.lookup_symbol(op, op.rewriter) is None:
            raise VerifyException(
                f"references an unknown rewriter symbol: {op.rewriter}"
            )


@irdl_op_definition
class SuccessOp(IRDLOperation):
    """Record a candidate rewrite and its captures, without terminating matching."""

    name = "match.success"

    rewriter = prop_def(SymbolRefAttr)
    benefit = prop_def(IntegerAttr.constr(i16, value=AtLeast(0)))
    inputs = var_operand_def(AnyPDLTypeConstr | base(RangeType[AnyPDLType]))

    traits = traits_def(HasAncestor(MatcherOp), SuccessOpSymbolUserOpInterface())

    assembly_format = (
        "$rewriter `benefit` `(` $benefit `)` "
        "(`(` $inputs^ `:` type($inputs) `)`)? attr-dict"
    )

    def __init__(
        self,
        rewriter: str | SymbolRefAttr,
        benefit: int | IntegerAttr,
        inputs: Sequence[SSAValue] = (),
    ):
        if isinstance(rewriter, str):
            rewriter = SymbolRefAttr(rewriter)
        if isinstance(benefit, int):
            benefit = IntegerAttr(benefit, i16)
        super().__init__(
            operands=[inputs], properties={"rewriter": rewriter, "benefit": benefit}
        )


@irdl_op_definition
class HasNameOp(IRDLOperation):
    """Fail the enclosing scope if the operation has a different name."""

    name = "match.has_name"

    input_op = operand_def(OperationType)
    operation_name = prop_def(StringAttr, prop_name="name")

    assembly_format = "$input_op `,` $name attr-dict"

    def __init__(self, input_op: SSAValue, name: str | StringAttr):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(operands=[input_op], properties={"name": name})


@irdl_op_definition
class GetOperandOp(IRDLOperation):
    """Get an operand, or a null handle if the index is out of bounds."""

    name = "match.get_operand"
    input_op = operand_def(OperationType)
    index = prop_def(IntegerAttr[I32])
    # The reference constrains only the wrapper, not its inner type.
    result = result_def(OptionalType)
    traits = traits_def(Pure())

    assembly_format = "$index `of` $input_op attr-dict `:` type($result)"

    def __init__(
        self,
        index: int | IntegerAttr[I32],
        input_op: SSAValue,
        result_type: OptionalType = OptionalType(ValueType()),
    ):
        if isinstance(index, int):
            index = IntegerAttr(index, i32)
        super().__init__(
            operands=[input_op], properties={"index": index}, result_types=[result_type]
        )


@irdl_op_definition
class GetResultOp(IRDLOperation):
    """Get a result, or a null handle if the index is out of bounds."""

    name = "match.get_result"
    input_op = operand_def(OperationType)
    index = prop_def(IntegerAttr[I32])
    result = result_def(OptionalType)
    traits = traits_def(Pure())

    assembly_format = "$index `of` $input_op attr-dict `:` type($result)"

    def __init__(
        self,
        index: int | IntegerAttr[I32],
        input_op: SSAValue,
        result_type: OptionalType = OptionalType(ValueType()),
    ):
        if isinstance(index, int):
            index = IntegerAttr(index, i32)
        super().__init__(
            operands=[input_op], properties={"index": index}, result_types=[result_type]
        )


@irdl_op_definition
class GetAttributeOp(IRDLOperation):
    """Get a named attribute, or a null handle if it is absent."""

    name = "match.get_attribute"
    input_op = operand_def(OperationType)
    attribute_name = prop_def(StringAttr, prop_name="name")
    result = result_def(OptionalType)
    traits = traits_def(Pure())

    assembly_format = "$name `of` $input_op attr-dict `:` type($result)"

    def __init__(
        self,
        name: str | StringAttr,
        input_op: SSAValue,
        result_type: OptionalType = OptionalType(AttributeType()),
    ):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(
            operands=[input_op], properties={"name": name}, result_types=[result_type]
        )


@irdl_op_definition
class GetDefiningOpOp(IRDLOperation):
    """Get the defining operation of a value (or the first value of a range)."""

    name = "match.get_defining_op"
    value = operand_def(ValueType | RangeType[ValueType])
    result = result_def(OptionalType)
    traits = traits_def(Pure())

    assembly_format = "`of` $value `:` type($value) attr-dict `->` type($result)"

    def __init__(
        self,
        value: SSAValue,
        result_type: OptionalType = OptionalType(OperationType()),
    ):
        super().__init__(operands=[value], result_types=[result_type])


@irdl_op_definition
class IsNotNullOp(IRDLOperation):
    """Unwrap a handle, failing the enclosing scope when it is null."""

    name = "match.is_not_null"
    T: ClassVar = VarConstraint("T", AnyPDLTypeConstr | base(RangeType[AnyPDLType]))
    optional_value = operand_def(ParamAttrConstraint(OptionalType, (T,)))
    unwrapped = result_def(T)

    assembly_format = "$optional_value `:` type($unwrapped) attr-dict"

    def __init__(self, optional_value: SSAValue[OptionalType]):
        super().__init__(
            operands=[optional_value], result_types=[optional_value.type.inner_type]
        )


@irdl_op_definition
class CheckOperandCountOp(IRDLOperation):
    """Fail unless the operation has exactly (or at least) the given operand count."""

    name = "match.check_operand_count"
    input_op = operand_def(OperationType)
    count = prop_def(IntegerAttr.constr(i32, value=AtLeast(0)))
    at_least = opt_prop_def(UnitAttr, prop_name="atLeast")

    assembly_format = "$input_op `is` (`at_least` $atLeast^)? $count attr-dict"

    def __init__(
        self, input_op: SSAValue, count: int | IntegerAttr[I32], at_least: bool = False
    ):
        if isinstance(count, int):
            count = IntegerAttr(count, i32)
        super().__init__(
            operands=[input_op],
            properties={"count": count, "atLeast": UnitAttr() if at_least else None},
        )


@irdl_op_definition
class CheckResultCountOp(IRDLOperation):
    """Fail unless the operation has exactly (or at least) the given result count."""

    name = "match.check_result_count"
    input_op = operand_def(OperationType)
    count = prop_def(IntegerAttr.constr(i32, value=AtLeast(0)))
    at_least = opt_prop_def(UnitAttr, prop_name="atLeast")

    assembly_format = "$input_op `is` (`at_least` $atLeast^)? $count attr-dict"

    def __init__(
        self, input_op: SSAValue, count: int | IntegerAttr[I32], at_least: bool = False
    ):
        if isinstance(count, int):
            count = IntegerAttr(count, i32)
        super().__init__(
            operands=[input_op],
            properties={"count": count, "atLeast": UnitAttr() if at_least else None},
        )


@irdl_op_definition
class HasAttrValueOp(IRDLOperation):
    """Fail unless the attribute equals the given constant."""

    name = "match.has_attr_value"
    attribute = operand_def(AttributeType)
    value = prop_def()

    assembly_format = "$attribute `is` $value attr-dict"

    def __init__(self, attribute: SSAValue, value: Attribute):
        super().__init__(operands=[attribute], properties={"value": value})


Match = Dialect(
    "match",
    [
        MatcherOp,
        SuccessOp,
        HasNameOp,
        GetOperandOp,
        GetResultOp,
        GetAttributeOp,
        GetDefiningOpOp,
        IsNotNullOp,
        CheckOperandCountOp,
        CheckResultCountOp,
        HasAttrValueOp,
    ],
    [OptionalType],
)
