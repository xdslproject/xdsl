from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import ClassVar

from xdsl.dialects.builtin import (
    I1,
    I32,
    I64,
    AnyFloatConstr,
    ArrayAttr,
    DenseArrayBase,
    DictionaryAttr,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    SignlessIntegerConstraint,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    VectorType,
    i1,
    i32,
    i64,
)
from xdsl.dialects.utils import FastMathAttrBase, FastMathFlag
from xdsl.ir import (
    Attribute,
    BitEnumAttribute,
    Dialect,
    EnumAttribute,
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
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    IsTerminator,
    NoMemoryEffect,
    Pure,
    SameOperandsAndResultType,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.str_enum import StrEnum

GEP_USE_SSA_VAL = -2147483648
"""

This is used in the getelementptr index list to signify that an ssa value
should be used for this index.

"""


def _parse_optional_llvm_type(parser: AttrParser) -> Attribute | None:
    """
    Used to parse llvm types without the `llvm.` prefix.
    """
    if parser.parse_optional_characters("void"):
        return LLVMVoidType()
    if parser.parse_optional_characters("ptr"):
        return LLVMPointerType(*LLVMPointerType.parse_parameters(parser))
    if parser.parse_optional_characters("array"):
        return LLVMArrayType(*LLVMArrayType.parse_parameters(parser))
    if parser.parse_optional_characters("struct"):
        return LLVMStructType(*LLVMStructType.parse_parameters(parser))


def parse_llvm_type(parser: AttrParser) -> Attribute:
    if (l := _parse_optional_llvm_type(parser)) is not None:
        return l
    return parser.parse_attribute()


def parse_optional_llvm_type(parser: AttrParser) -> Attribute | None:
    if (l := _parse_optional_llvm_type(parser)) is not None:
        return l
    return parser.parse_optional_attribute()


@irdl_attr_definition
class LLVMStructType(ParametrizedAttribute, TypeAttribute):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types).
    """

    name = "llvm.struct"

    # An empty string refers to a struct without a name.
    struct_name: StringAttr
    types: ArrayAttr[Attribute]

    # TODO: Add this parameter once xDSL supports the necessary capabilities.
    #  bitmask: StringAttr

    @staticmethod
    def from_type_list(types: Sequence[Attribute]) -> LLVMStructType:
        return LLVMStructType(StringAttr(""), ArrayAttr(types))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if self.struct_name.data:
                printer.print_string_literal(self.struct_name.data)
                printer.print_string(", ")
            with printer.in_parens():
                printer.print_list(self.types.data, printer.print_attribute)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[StringAttr, ArrayAttr]:
        parser.parse_characters("<", " in LLVM struct")
        struct_name = parser.parse_optional_str_literal()
        if struct_name is None:
            struct_name = ""
        else:
            parser.parse_characters(",", " after type")

        params = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_llvm_type(parser)
        )
        parser.parse_characters(">", " to close LLVM struct parameters")
        return (StringAttr(struct_name), ArrayAttr(params))


@irdl_attr_definition
class LLVMPointerType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.ptr"

    addr_space: IntAttr | NoneAttr

    def print_parameters(self, printer: Printer) -> None:
        if isinstance(self.addr_space, IntAttr):
            with printer.in_angle_brackets():
                printer.print_int(self.addr_space.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr | NoneAttr]:
        if parser.parse_optional_characters("<") is None:
            return (NoneAttr(),)
        addr_space = parser.parse_integer()
        parser.parse_characters(">", " to end llvm.ptr parameters")
        return (IntAttr(addr_space),)

    def __init__(self, addr_space: IntAttr | NoneAttr = NoneAttr()):
        super().__init__(addr_space)


@irdl_attr_definition
class LLVMArrayType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.array"

    size: IntAttr
    type: Attribute

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.size.data)
            printer.print_string(" x ")
            printer.print_attribute(self.type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr, Attribute]:
        with parser.in_angle_brackets():
            size = IntAttr(parser.parse_integer())
            parser.parse_shape_delimiter()
            type = parse_llvm_type(parser)
        return (size, type)

    @staticmethod
    def from_size_and_type(size: int | IntAttr, type: Attribute):
        if isinstance(size, int):
            size = IntAttr(size)
        return LLVMArrayType(size, type)


@irdl_attr_definition
class LLVMVoidType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.void"


@irdl_attr_definition
class LLVMFunctionType(ParametrizedAttribute, TypeAttribute):
    """
    Currently does not support variadics.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#function-types).
    """

    name = "llvm.func"

    inputs: ArrayAttr[Attribute]
    output: Attribute
    variadic: UnitAttr | NoneAttr

    def __init__(
        self,
        inputs: Sequence[Attribute] | ArrayAttr[Attribute],
        output: Attribute | None = None,
        is_variadic: bool = False,
    ) -> None:
        if not isinstance(inputs, ArrayAttr):
            inputs = ArrayAttr(inputs)
        if output is None:
            output = LLVMVoidType()
        variad_attr = UnitAttr() if is_variadic else NoneAttr()
        super().__init__(inputs, output, variad_attr)

    @property
    def is_variadic(self) -> bool:
        return isinstance(self.variadic, UnitAttr)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if isinstance(self.output, LLVMVoidType):
                printer.print_string("void")
            else:
                printer.print_attribute(self.output)

            printer.print_string(" ")
            with printer.in_parens():
                printer.print_list(self.inputs, printer.print_attribute)
                if self.is_variadic:
                    if self.inputs:
                        printer.print_string(", ")
                    printer.print_string("...")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<", " in llvm.func parameters")
        output = parse_llvm_type(parser)

        # save pos before args for error message printing
        pos = parser.pos

        def _parse_attr_or_variadic() -> Attribute | EllipsisType:
            """
            This returns either an attribute, or Ellipsis if a
            varargs specifier (`...`) was parsed.
            """
            if parser.parse_optional_characters("...") is not None:
                return ...
            return parse_llvm_type(parser)

        inputs = parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, _parse_attr_or_variadic
        )
        is_varargs: NoneAttr | UnitAttr = NoneAttr()
        if inputs and inputs[-1] is Ellipsis:
            is_varargs = UnitAttr()
            inputs = inputs[:-1]

        if not isa(inputs, list[Attribute]):
            parser.raise_error(
                "Varargs specifier `...` must be at the end of the argument definition",
                pos,
                parser.pos,
            )

        parser.parse_characters(">", " in llvm.func parameters")

        return [ArrayAttr(inputs), output, is_varargs]


@irdl_attr_definition
class LinkageAttr(ParametrizedAttribute):
    name = "llvm.linkage"

    linkage: StringAttr

    def __init__(self, linkage: str | StringAttr) -> None:
        if isinstance(linkage, str):
            linkage = StringAttr(linkage)
        super().__init__(linkage)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.linkage)
        printer.print_string(">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<", "llvm.linkage parameter expected")
        # The linkage string is output from xDSL as a string (and accepted by MLIR as such)
        # however it is always output from MLIR without quotes. Therefore need to determine
        # whether this is a string or not and slightly change how we parse based upon that
        linkage_str = parser.parse_optional_str_literal()
        if linkage_str is None:
            linkage_str = parser.parse_identifier()
        linkage = StringAttr(linkage_str)
        parser.parse_characters(">", " to end llvm.linkage parameters")
        return [linkage]

    def verify(self):
        allowed_linkage = [
            "private",
            "internal",
            "available_externally",
            "linkonce",
            "weak",
            "common",
            "appending",
            "extern_weak",
            "linkonce_odr",
            "weak_odr",
            "external",
        ]
        if self.linkage.data not in allowed_linkage:
            raise VerifyException(f"Specified linkage '{self.linkage.data}' is unknown")


class ArithmeticBinOperation(IRDLOperation, ABC):
    """Class for arithmetic binary operations."""

    T: ClassVar = VarConstraint(
        "T", SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[lhs.type],
        )

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_unresolved_operand()
        parser.parse_characters(",")
        rhs = parser.parse_unresolved_operand()
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        type = parser.parse_type()
        operands = parser.resolve_operands([lhs, rhs], [type, type], parser.pos)
        return cls(operands[0], operands[1], attributes)

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.lhs)
        printer.print_string(", ")
        printer.print_ssa_value(self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.lhs.type)


class OverflowFlag(StrEnum):
    NO_SIGNED_WRAP = "nsw"
    NO_UNSIGNED_WRAP = "nuw"


@dataclass(frozen=True, init=False)
class OverflowAttrBase(BitEnumAttribute[OverflowFlag]):
    none_value = "none"


@irdl_attr_definition(init=False)
class OverflowAttr(OverflowAttrBase):
    name = "llvm.overflow"

    @classmethod
    def parse(cls, parser: Parser) -> OverflowAttr:
        if parser.parse_optional_keyword("overflow") is not None:
            return OverflowAttr(OverflowAttr.parse_parameter(parser))
        return OverflowAttr("none")

    def print(self, printer: Printer):
        if self.flags:
            printer.print_string(" overflow")
            self.print_parameter(printer)

    def to_int(self) -> int:
        if len(self.data) == 0:
            return 0
        if len(self.data) == 2:
            return 3
        if self.data[0] == OverflowFlag.NO_SIGNED_WRAP:
            return 1
        return 2

    @staticmethod
    def from_int(i: int) -> OverflowAttr:
        match i:
            case 0:
                return OverflowAttr("none")
            case 1:
                return OverflowAttr((OverflowFlag.NO_SIGNED_WRAP,))
            case 2:
                return OverflowAttr((OverflowFlag.NO_UNSIGNED_WRAP,))
            case 3:
                return OverflowAttr(
                    (OverflowFlag.NO_SIGNED_WRAP, OverflowFlag.NO_UNSIGNED_WRAP)
                )
            case _:
                raise ValueError("OverflowAttr given out of bounds integer.")


class ArithmeticBinOpOverflow(IRDLOperation, ABC):
    """Class for arithmetic binary operations that use overflow flags."""

    T: ClassVar = VarConstraint(
        "T", SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)
    overflowFlags = opt_prop_def(IntegerAttr[I32])

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
        overflow: OverflowAttr | IntegerAttr = IntegerAttr(0, 32),
    ):
        if isinstance(overflow, OverflowAttr):
            overflow = IntegerAttr(overflow.to_int(), 32)
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[lhs.type],
            properties={
                "overflowFlags": overflow,
            },
        )

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_unresolved_operand()
        parser.parse_characters(",")
        rhs = parser.parse_unresolved_operand()
        overflowFlags = OverflowAttr.parse(parser)
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        type = parser.parse_type()
        operands = parser.resolve_operands([lhs, rhs], [type, type], parser.pos)
        return cls(operands[0], operands[1], attributes, overflowFlags)

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.lhs)
        printer.print_string(", ")
        printer.print_ssa_value(self.rhs)
        if self.overflowFlags:
            OverflowAttr.from_int(self.overflowFlags.value.data).print(printer)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.lhs.type)


class ArithmeticBinOpExact(IRDLOperation, ABC):
    """Class for arithmetic binary operations that use an exact flag."""

    T: ClassVar = VarConstraint(
        "T", SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)
    is_exact = opt_prop_def(UnitAttr, prop_name="isExact")

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
        is_exact: UnitAttr | None = None,
    ):
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[lhs.type],
            properties={
                "isExact": is_exact,
            },
        )

    @classmethod
    def parse_exact(cls, parser: Parser):
        if parser.parse_optional_keyword("exact") is not None:
            return UnitAttr()

    def print_exact(self, printer: Printer) -> None:
        if self.is_exact:
            printer.print_string(" exact")

    @classmethod
    def parse(cls, parser: Parser):
        exact = cls.parse_exact(parser)
        lhs = parser.parse_unresolved_operand()
        parser.parse_characters(",")
        rhs = parser.parse_unresolved_operand()
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        type = parser.parse_type()
        operands = parser.resolve_operands([lhs, rhs], [type, type], parser.pos)
        return cls(operands[0], operands[1], attributes, exact)

    def print(self, printer: Printer) -> None:
        self.print_exact(printer)
        printer.print_string(" ")
        printer.print_ssa_value(self.lhs)
        printer.print_string(", ")
        printer.print_ssa_value(self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.lhs.type)


class ArithmeticBinOpDisjoint(IRDLOperation, ABC):
    """Class for arithmetic binary operations that use a disjoint flag."""

    T: ClassVar = VarConstraint(
        "T", SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)
    is_disjoint = opt_prop_def(UnitAttr, prop_name="isDisjoint")

    traits = traits_def(NoMemoryEffect())

    assembly_format = (
        "(`disjoint` $isDisjoint^)? $lhs `,` $rhs attr-dict `:` type($res)"
    )

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
        is_disjoint: UnitAttr | None = None,
    ):
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[lhs.type],
            properties={
                "isDisjoint": is_disjoint,
            },
        )


class IntegerConversionOp(IRDLOperation, ABC):
    arg = operand_def(
        SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    res = result_def(
        SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        arg: SSAValue,
        res_type: Attribute,
        attributes: dict[str, Attribute] = {},
    ):
        super().__init__(operands=[arg], attributes=attributes, result_types=[res_type])

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_unresolved_operand()
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        arg_type = parser.parse_type()
        parser.parse_characters("to")
        res_type = parser.parse_type()
        operands = parser.resolve_operands([arg], [arg_type], parser.pos)
        return cls(operands[0], res_type, attributes)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.arg)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.arg.type)
        printer.print_string(" to ")
        printer.print_attribute(self.res.type)


class IntegerConversionOpNNeg(IRDLOperation, ABC):
    arg = operand_def(
        SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )
    res = result_def(
        SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )
    traits = traits_def(NoMemoryEffect())
    non_neg = opt_prop_def(UnitAttr, prop_name="nonNeg")

    assembly_format = "(`nneg` $nonNeg^)? $arg attr-dict `:` type($arg) `to` type($res)"

    def __init__(
        self,
        arg: SSAValue,
        res_type: Attribute,
        attributes: dict[str, Attribute] = {},
        non_neg: UnitAttr | None = None,
    ):
        super().__init__(
            operands=(arg,),
            attributes=attributes,
            result_types=(res_type,),
            properties={
                "nonNeg": non_neg,
            },
        )


class IntegerConversionOpOverflow(IRDLOperation, ABC):
    arg = operand_def(
        SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )
    res = result_def(
        SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )
    overflowFlags = opt_prop_def(OverflowAttr)
    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        arg: SSAValue,
        res_type: Attribute,
        attributes: dict[str, Attribute] = {},
        overflow: OverflowAttr = OverflowAttr(None),
    ):
        super().__init__(
            operands=(arg,),
            attributes=attributes,
            result_types=(res_type,),
            properties={
                "overflowFlags": overflow,
            },
        )

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_unresolved_operand()
        overflowFlags = OverflowAttr.parse(parser)
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        arg_type = parser.parse_type()
        parser.parse_characters("to")
        res_type = parser.parse_type()
        operands = parser.resolve_operands([arg], [arg_type], parser.pos)
        return cls(operands[0], res_type, attributes, overflowFlags)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.arg)
        if self.overflowFlags:
            self.overflowFlags.print(printer)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.arg.type)
        printer.print_string(" to ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class AddOp(ArithmeticBinOpOverflow):
    name = "llvm.add"


@irdl_op_definition
class SubOp(ArithmeticBinOpOverflow):
    name = "llvm.sub"


@irdl_op_definition
class MulOp(ArithmeticBinOpOverflow):
    name = "llvm.mul"


@irdl_op_definition
class UDivOp(ArithmeticBinOpExact):
    name = "llvm.udiv"


@irdl_op_definition
class SDivOp(ArithmeticBinOpExact):
    name = "llvm.sdiv"


@irdl_op_definition
class URemOp(ArithmeticBinOperation):
    name = "llvm.urem"


@irdl_op_definition
class SRemOp(ArithmeticBinOperation):
    name = "llvm.srem"


@irdl_op_definition
class AndOp(ArithmeticBinOperation):
    name = "llvm.and"


@irdl_op_definition
class OrOp(ArithmeticBinOpDisjoint):
    name = "llvm.or"


@irdl_op_definition
class XOrOp(ArithmeticBinOperation):
    name = "llvm.xor"


@irdl_op_definition
class ShlOp(ArithmeticBinOpOverflow):
    name = "llvm.shl"


@irdl_op_definition
class LShrOp(ArithmeticBinOpExact):
    name = "llvm.lshr"


@irdl_op_definition
class AShrOp(ArithmeticBinOpExact):
    name = "llvm.ashr"


@irdl_op_definition
class TruncOp(IntegerConversionOpOverflow):
    name = "llvm.trunc"

    def verify(self, verify_nested_ops: bool = True):
        arg_type = (
            arg_t.element_type if isa(arg_t := self.arg.type, VectorType) else arg_t
        )
        res_type = (
            res_t.element_type if isa(res_t := self.res.type, VectorType) else res_t
        )

        assert isa(arg_type, IntegerType)
        assert isa(res_type, IntegerType)

        if arg_type.bitwidth <= res_type.bitwidth:
            raise VerifyException(
                f"invalid cast opcode for cast from {arg_type} to {res_type}"
            )
        super().verify(verify_nested_ops)


@irdl_op_definition
class ZExtOp(IntegerConversionOpNNeg):
    name = "llvm.zext"

    def verify(self, verify_nested_ops: bool = True):
        arg_type = (
            arg_t.element_type if isa(arg_t := self.arg.type, VectorType) else arg_t
        )
        res_type = (
            res_t.element_type if isa(res_t := self.res.type, VectorType) else res_t
        )

        assert isa(arg_type, IntegerType)
        assert isa(res_type, IntegerType)
        if arg_type.bitwidth >= res_type.bitwidth:
            raise VerifyException(
                f"invalid cast opcode for cast from {arg_type} to {res_type}"
            )
        super().verify(verify_nested_ops)


@irdl_op_definition
class SExtOp(IntegerConversionOp):
    name = "llvm.sext"

    def verify(self, verify_nested_ops: bool = True):
        arg_type = (
            arg_t.element_type if isa(arg_t := self.arg.type, VectorType) else arg_t
        )
        res_type = (
            res_t.element_type if isa(res_t := self.res.type, VectorType) else res_t
        )

        assert isa(arg_type, IntegerType)
        assert isa(res_type, IntegerType)
        if arg_type.bitwidth >= res_type.bitwidth:
            raise VerifyException(
                f"invalid cast opcode for cast from {arg_type} to {res_type}"
            )
        super().verify(verify_nested_ops)


class ICmpPredicateFlag(StrEnum):
    EQ = "eq"
    NE = "ne"
    SLT = "slt"
    SLE = "sle"
    SGT = "sgt"
    SGE = "sge"
    ULT = "ult"
    ULE = "ule"
    UGT = "ugt"
    UGE = "uge"

    @staticmethod
    def from_int(index: int) -> ICmpPredicateFlag:
        return ALL_ICMP_FLAGS[index]

    def to_int(self) -> int:
        return ICMP_INDEX_BY_FLAG[self]


ALL_ICMP_FLAGS = tuple(ICmpPredicateFlag)
ICMP_INDEX_BY_FLAG = {f: i for (i, f) in enumerate(ALL_ICMP_FLAGS)}


@irdl_op_definition
class ICmpOp(IRDLOperation):
    name = "llvm.icmp"
    T: ClassVar = VarConstraint(
        "T", SignlessIntegerConstraint | VectorType.constr(SignlessIntegerConstraint)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(I1 | VectorType[I1])
    predicate = prop_def(IntegerAttr[i64])

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        predicate: IntegerAttr[IntegerType],
        attributes: dict[str, Attribute] = {},
    ):
        result_type = (
            VectorType(i1, lhs_type.shape)
            if isa(lhs_type := lhs.type, VectorType)
            else i1
        )

        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[result_type],
            properties={
                "predicate": predicate,
            },
        )

    @classmethod
    def parse(cls, parser: Parser):
        predicate_literal = parser.parse_str_literal()
        predicate_value = ICmpPredicateFlag[predicate_literal.upper()]
        predicate_int = predicate_value.to_int()
        predicate = IntegerAttr(predicate_int, i64)
        lhs = parser.parse_unresolved_operand()
        parser.parse_characters(",")
        rhs = parser.parse_unresolved_operand()
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        type = parser.parse_type()
        operands = parser.resolve_operands([lhs, rhs], [type, type], parser.pos)
        return cls(operands[0], operands[1], predicate, attributes)

    def print_predicate(self, printer: Printer):
        flag = ICmpPredicateFlag.from_int(self.predicate.value.data)
        printer.print_string(f"{flag}")

    def print(self, printer: Printer):
        printer.print_string(' "')
        self.print_predicate(printer)
        printer.print_string('" ')
        printer.print_ssa_value(self.lhs)
        printer.print_string(", ")
        printer.print_ssa_value(self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.lhs.type)

    def verify_(self, verify_nested_ops: bool = True) -> None:
        if isa(self.lhs.type, VectorType):
            if not isa(res_type := self.res.type, VectorType):
                raise VerifyException(
                    f"Result must be a vector if operands are vectors, got {res_type}"
                )
        else:
            if isa(res_type := self.res.type, VectorType):
                raise VerifyException(
                    f"Result must be scalar if operands are scalar, got {res_type}"
                )


@irdl_op_definition
class GEPOp(IRDLOperation):
    """
    llvm.getelementptr is an instruction to do pointer arithmetic by
    adding/subtracting offsets from a pointer.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmgetelementptr-mlirllvmgepop).

    GetElementPtr is documented in various places online:

    See the LLVM [documentation](https://www.llvm.org/docs/GetElementPtr.html).
    A good [blogpost](https://blog.yossarian.net/2020/09/19/LLVMs-getelementptr-by-example).

    Note that the first two discuss *LLVM IRs* GEP operation, not the MLIR one.
    The semantics are the same, but the structure used by MLIR is not well
    documented (yet) and the syntax is a bit different.

    Here we focus on MLIRs GEP operation:

    %res = llvm.getelementptr %ptr  [1, 2, %val]
                              ^^^^   ^^^^^^^^^^
                              input   indices

    The central point to understanding GEP is that:
    > GEP never dereferences, it only does math on the given pointer

    It *always* returns a pointer to the element "selected" that is some
    number of bytes offset from the input pointer:

    `result = ptr + x` for some x parametrized by the arguments

    ## Examples:

    Given the following pointer:

    %ptr : llvm.ptr<llvm.struct<(i32, i32, llvm.array<2xi32>)>>

    The following indices point to the following things:

    [0]      -> The first element of the pointer, so a pointer to the struct:
                llvm.ptr<llvm.struct<(i32, i32, llvm.array<2xi32>)>>

    [1]      -> The *next* element of the pointer, useful if the
                pointer points to a list of structs.
                Equivalent to (ptr + 1), so points to
                llvm.ptr<llvm.struct<(i32, i32, llvm.array<2xi32>)>>

    [0,0]    -> The first member of the first struct:
                llvm.ptr<i32>

    [1,0]    -> The first member of the *second* struct pointed to by ptr
                (can result in out-of-bounds access if the ptr only points to a single struct)
                llvm.ptr<i32>

    [0,2]    -> The third member of the first struct.
                llvm.ptr<llvm.array<2,i32>>

    [0,2,0]  -> The first entry of the array that is the third member of
                the first struct pointed to by our ptr.
                llvm.ptr<i32>

    [0,0,1]  -> Invalid! The first element of the first struct has no "sub-elements"!


    Here is an example of invalid GEP operation parameters:

    Given a different pointer to the example above:

    %ptr : llvm.ptr<llvm.struct<(llvm.ptr<i32>, i32)>>

    Note the two pointers, one to the struct, one in the struct.

    We can do math on the first pointer:

    [0]      -> First struct
                llvm.ptr<llvm.struct<(llvm.ptr<i32>, i32)>>

    [0,1]    -> Second member of first struct
                llvm.ptr<i32>

    [0,0]    -> First member of the first struct
                llvm.ptr<llvm.ptr<i32>>

    [0,0,3]  -> Invalid! In order to find the fourth element in the pointer
                it would need to be dereferenced! GEP can't do that!

    Expressed in "C", this would equate to:

    # address of first struct
    (ptr + 0)

    # address of first field of first struct
    &((ptr + 0)->elm0)
               ^^^^^^
               Even though it looks like it, we are not actually
               dereferencing ptr here.

    # address of fourth element:
    &(((ptr + 0)->elm0 + 3))
                ^^^^^^^^^^
                This actually dereferences (ptr + 0) to access elm0!

    Which translates to roughly this MLIR code:

    %elm0_addr   = llvm.gep %ptr[0,0]   : (!llvm.ptr<...>) -> !llvm.ptr<!llvm.ptr<i32>>
    %elm0        = llvm.load %elm0_addr : (!llvm.ptr<llvm.ptr<i32>>) -> !llvm.ptr<i32>
    %elm0_3_addr = llvm.gep %elm0[3]    : !llvm.ptr<i32> -> !llvm.ptr<i32>

    Here the necessary dereferencing is very visible, as %elm0_3_addr is only
    accessible through an `llvm.load` on %elm0_addr.
    """

    name = "llvm.getelementptr"

    ptr = operand_def(LLVMPointerType)
    ssa_indices = var_operand_def(IntegerType)
    elem_type = prop_def()
    noWrapFlags = prop_def(IntegerAttr[I32])

    result = result_def(LLVMPointerType)

    rawConstantIndices = prop_def(DenseArrayBase)
    inbounds = opt_prop_def(UnitAttr)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        ptr: SSAValue | Operation,
        indices: Sequence[int],
        pointee_type: Attribute,
        ssa_indices: Sequence[SSAValue | Operation] | None = None,
        result_type: LLVMPointerType = LLVMPointerType(),
        inbounds: bool = False,
    ):
        """
        A basic constructor for the GEPOp.

        Pass the GEP_USE_SSA_VAL magic value in place of each constant
        index that you want to be read from an SSA value.

        Take a look at `from_mixed_indices` for something without
        magic values.
        """
        if ssa_indices is None:
            ssa_indices = []

        props: dict[str, Attribute] = {
            "rawConstantIndices": DenseArrayBase.from_list(i32, indices),
            "elem_type": pointee_type,
            "noWrapFlags": IntegerAttr(0, 32),
        }

        props["elem_type"] = pointee_type

        if inbounds:
            props["inbounds"] = UnitAttr()

        super().__init__(
            operands=[ptr, ssa_indices], result_types=[result_type], properties=props
        )

    @staticmethod
    def from_mixed_indices(
        ptr: SSAValue | Operation,
        indices: Sequence[int | SSAValue | Operation],
        pointee_type: Attribute,
        result_type: LLVMPointerType = LLVMPointerType(),
        inbounds: bool = False,
    ):
        """
        This is a helper function that accepts a mixed list of SSA values and const
        indices. It will automatically construct the correct indices and ssa_indices
        lists from that.

        You can call this using [1, 2, some_ssa_val, 3] as the indices array.

        Other than that, this behaves exactly the same as `.get`
        """
        ssa_indices: list[SSAValue] = []
        const_indices: list[int] = []
        for idx in indices:
            if isinstance(idx, int):
                const_indices.append(idx)
            else:
                const_indices.append(GEP_USE_SSA_VAL)
                ssa_indices.append(SSAValue.get(idx))
        return GEPOp(
            ptr,
            const_indices,
            pointee_type,
            ssa_indices,
            result_type=result_type,
            inbounds=inbounds,
        )


@irdl_op_definition
class AllocaOp(IRDLOperation):
    name = "llvm.alloca"

    size = operand_def(IntegerType)

    alignment = opt_prop_def(IntegerAttr)
    elem_type = opt_prop_def()

    res = result_def()

    def __init__(
        self,
        size: SSAValue | Operation,
        elem_type: Attribute,
        alignment: int = 32,
    ):
        props: dict[str, Attribute] = {
            "alignment": IntegerAttr.from_int_and_width(alignment, 64)
        }
        ptr_type = LLVMPointerType()
        props["elem_type"] = elem_type

        super().__init__(operands=[size], properties=props, result_types=[ptr_type])


@irdl_op_definition
class IntToPtrOp(IRDLOperation):
    name = "llvm.inttoptr"

    input = operand_def(IntegerType)

    output = result_def(LLVMPointerType)

    traits = traits_def(NoMemoryEffect())

    def __init__(self, input: SSAValue | Operation):
        ptr_type = LLVMPointerType()
        super().__init__(operands=[input], result_types=[ptr_type])


class TailCallKind(StrEnum):
    NONE = "none"
    TAIL = "tail"
    MUST_TAIL = "musttail"
    NOTAIL = "notail"


@irdl_attr_definition
class TailCallKindAttr(EnumAttribute[TailCallKind]):
    name = "llvm.tailcallkind"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> TailCallKind:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)


@irdl_op_definition
class InlineAsmOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminline_asm-llvminlineasmop).

    To see what each field means, have a look [here](https://llvm.org/docs/LangRef.html#inline-assembler-expressions).
    """

    name = "llvm.inline_asm"

    operands_ = var_operand_def()

    res = opt_result_def()

    # note: in MLIR upstream this is implemented as AsmDialectAttr;
    # which is an instantiation of an LLVM_EnumAttr
    # 0 for AT&T inline assembly dialect
    # 1 for Intel inline assembly dialect
    # In this context dialect does not refer to an MLIR dialect
    asm_dialect = opt_prop_def(IntegerAttr[I64])

    asm_string = prop_def(StringAttr)
    constraints = prop_def(StringAttr)

    has_side_effects = opt_prop_def(UnitAttr)
    is_align_stack = opt_prop_def(UnitAttr)

    tail_call_kind = prop_def(
        TailCallKindAttr, default_value=TailCallKindAttr(TailCallKind.NONE)
    )

    def __init__(
        self,
        asm_string: str,
        constraints: str,
        operands: Sequence[SSAValue | Operation],
        res_types: Sequence[Attribute] | None = None,
        asm_dialect: int = 0,
        has_side_effects: bool = False,
        is_align_stack: bool = False,
        tail_call_kind: TailCallKindAttr | None = None,
    ):
        props: dict[str, Attribute | None] = {
            "asm_string": StringAttr(asm_string),
            "constraints": StringAttr(constraints),
            "asm_dialect": IntegerAttr.from_int_and_width(asm_dialect, 64),
            "has_side_effects": UnitAttr() if has_side_effects else None,
            "is_align_stack": UnitAttr() if is_align_stack else None,
            "tail_call_kind": tail_call_kind,
        }

        if res_types is None:
            res_types = []

        super().__init__(
            operands=[operands],
            properties=props,
            result_types=[res_types],
        )


@irdl_op_definition
class PtrToIntOp(IRDLOperation):
    name = "llvm.ptrtoint"

    input = operand_def(LLVMPointerType)

    output = result_def(IntegerType)

    traits = traits_def(NoMemoryEffect())

    def __init__(self, arg: SSAValue | Operation, int_type: Attribute = i64):
        super().__init__(operands=[arg], result_types=[int_type])


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "llvm.load"

    ptr = operand_def(LLVMPointerType)

    alignment = opt_prop_def(IntegerAttr[IntegerType])
    ordering = prop_def(IntegerAttr[IntegerType], default_value=IntegerAttr(0, i64))

    dereferenced_value = result_def()

    def __init__(
        self,
        ptr: SSAValue | Operation,
        result_type: Attribute,
        alignment: int | None = None,
        ordering: int = 0,
    ):
        props: dict[str, Attribute] = {
            "ordering": IntegerAttr(ordering, i64),
        }

        if alignment is not None:
            props["alignment"] = IntegerAttr(alignment, i64)

        super().__init__(operands=[ptr], result_types=[result_type], properties=props)


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "llvm.store"

    value = operand_def()
    ptr = operand_def(LLVMPointerType)

    alignment = opt_prop_def(IntegerAttr[IntegerType])
    ordering = opt_prop_def(IntegerAttr[IntegerType])
    volatile_ = opt_prop_def(UnitAttr)
    nontemporal = opt_prop_def(UnitAttr)

    def __init__(
        self,
        value: SSAValue | Operation,
        ptr: SSAValue | Operation,
        alignment: int | None = None,
        ordering: int = 0,
        volatile: bool = False,
        nontemporal: bool = False,
    ):
        props: dict[str, Attribute] = {
            "ordering": IntegerAttr(ordering, i64),
        }

        if alignment is not None:
            props["alignment"] = IntegerAttr[IntegerType](alignment, i64)
        if volatile:
            props["volatile_"] = UnitAttr()
        if nontemporal:
            props["nontemporal"] = UnitAttr()

        super().__init__(
            operands=[value, ptr],
            properties=props,
            result_types=[],
        )


@irdl_op_definition
class NullOp(IRDLOperation):
    name = "llvm.mlir.null"

    nullptr = result_def(LLVMPointerType)

    traits = traits_def(NoMemoryEffect())

    def __init__(self, ptr_type: LLVMPointerType | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType()

        super().__init__(result_types=[ptr_type])


@irdl_op_definition
class ExtractValueOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmextractvalue-mlirllvmextractvalueop).
    """

    name = "llvm.extractvalue"

    position = prop_def(DenseArrayBase.constr(i64))
    container = operand_def(Attribute)

    res = result_def(Attribute)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        position: DenseArrayBase,
        container: SSAValue | Operation,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[container],
            properties={
                "position": position,
            },
            result_types=[result_type],
        )


@irdl_op_definition
class InsertValueOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminsertvalue-mlirllvminsertvalueop).
    """

    name = "llvm.insertvalue"

    position = prop_def(DenseArrayBase.constr(i64))
    container = operand_def(Attribute)
    value = operand_def(Attribute)

    res = result_def(Attribute)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        position: DenseArrayBase,
        container: SSAValue,
        value: SSAValue,
    ):
        super().__init__(
            operands=[container, value],
            properties={
                "position": position,
            },
            result_types=[container.type],
        )


@irdl_op_definition
class UndefOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirundef-mlirllvmundefop).
    """

    name = "llvm.mlir.undef"

    res = result_def(Attribute)

    traits = traits_def(NoMemoryEffect())

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])


@irdl_op_definition
class GlobalOp(IRDLOperation):
    name = "llvm.mlir.global"

    global_type = prop_def()
    constant = opt_prop_def(UnitAttr)
    sym_name = prop_def(SymbolNameConstraint())
    linkage = prop_def(LinkageAttr)
    dso_local = opt_prop_def(UnitAttr)
    thread_local_ = opt_prop_def(UnitAttr)
    visibility_ = opt_prop_def(IntegerAttr[IntegerType])
    value = opt_prop_def()
    alignment = opt_prop_def(IntegerAttr)
    addr_space = prop_def(IntegerAttr)
    unnamed_addr = opt_prop_def(IntegerAttr)
    section = opt_prop_def(StringAttr)

    # This always needs an empty region as it is in the top level module definition
    body = region_def()

    traits = traits_def(SymbolOpInterface())

    def __init__(
        self,
        global_type: Attribute,
        sym_name: str | StringAttr,
        linkage: str | LinkageAttr,
        addr_space: int = 0,
        constant: bool | None = None,
        dso_local: bool | None = None,
        thread_local_: bool | None = None,
        value: Attribute | None = None,
        alignment: int | None = None,
        unnamed_addr: int | None = None,
        section: str | StringAttr | None = None,
        body: Region | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)

        if isinstance(linkage, str):
            linkage = LinkageAttr(linkage)

        props: dict[str, Attribute] = {
            "global_type": global_type,
            "sym_name": sym_name,
            "linkage": linkage,
            "addr_space": IntegerAttr(addr_space, 32),
        }

        if constant is not None and constant:
            props["constant"] = UnitAttr()

        if dso_local is not None and dso_local:
            props["dso_local"] = UnitAttr()

        if thread_local_ is not None and thread_local_:
            props["thread_local_"] = UnitAttr()

        if value is not None:
            props["value"] = value

        if alignment is not None:
            props["alignment"] = IntegerAttr(alignment, 64)

        if unnamed_addr is not None:
            props["unnamed_addr"] = IntegerAttr(unnamed_addr, 64)

        if section is not None:
            if isinstance(section, str):
                section = StringAttr(section)
            props["section"] = section

        if body is None:
            body = Region()

        super().__init__(properties=props, regions=(body,))


@irdl_op_definition
class AddressOfOp(IRDLOperation):
    name = "llvm.mlir.addressof"

    global_name = prop_def(SymbolRefAttr)
    result = result_def(LLVMPointerType)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        global_name: str | StringAttr | SymbolRefAttr,
        result_type: LLVMPointerType,
    ):
        if isinstance(global_name, StringAttr | str):
            global_name = SymbolRefAttr(global_name)

        super().__init__(
            properties={"global_name": global_name}, result_types=[result_type]
        )


LLVM_CALLING_CONVS: set[str] = {
    "ccc",
    "fastcc",
    "coldcc",
    "cc 10",
    "cc 11",
    "webkit_jscc",
    "anyregcc",
    "preserve_mostcc",
    "preserve_allcc",
    "cxx_fast_tlscc",
    "tailcc",
    "swiftcc",
    "swifttailcc",
    "cfguard_checkcc",
}
"""
A list of all valid calling conventions understood by LLVM, see external documentation
[here](https://llvm.org/docs/LangRef.html#calling-conventions) for more info.
"""


@irdl_attr_definition
class CallingConventionAttr(ParametrizedAttribute):
    """
    LLVM Calling convention, default is ccc.
    """

    name = "llvm.cconv"

    convention: StringAttr

    @property
    def cconv_name(self) -> str:
        return self.convention.data

    def __init__(self, conv: str):
        super().__init__(StringAttr(conv))

    def _verify(self):
        if self.cconv_name not in LLVM_CALLING_CONVS:
            raise VerifyException(f'Invalid calling convention "{self.cconv_name}"')

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<" + self.convention.data + ">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        for conv in LLVM_CALLING_CONVS:
            if parser.parse_optional_characters(conv) is not None:
                parser.parse_characters(">")
                return [StringAttr(conv)]
        parser.raise_error("Unknown calling convention")


class FramePointerKind(StrEnum):
    NONE = "none"
    NONLEAF = "non-leaf"
    ALL = "all"
    RESERVED = "reserved"


@irdl_attr_definition
class FramePointerKindAttr(EnumAttribute[FramePointerKind]):
    """LLVM Frame Pointer Kind."""

    name = "llvm.framePointerKind"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> FramePointerKind:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)


@irdl_attr_definition
class TargetFeaturesAttr(ParametrizedAttribute):
    """
    Represents the LLVM target features as a list that can be checked within
    passes/rewrites.
    """

    name = "llvm.target_features"

    features: ArrayAttr[StringAttr]

    def verify(self):
        for feature in self.features:
            if not feature.data.startswith(("-", "+")):
                raise VerifyException("target features must start with '+' or '-'")


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "llvm.func"

    body = region_def()
    sym_name = prop_def(SymbolNameConstraint())
    function_type = prop_def(LLVMFunctionType)
    CConv = prop_def(CallingConventionAttr)
    linkage = prop_def(LinkageAttr)
    sym_visibility = opt_prop_def(StringAttr)
    visibility_ = prop_def(IntegerAttr[IntegerType])

    # The following properties are not yet verified by the xDSL verifier, but
    # are verified to at least allow the IR to be parsed and printed correctly.
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    frame_pointer = opt_prop_def(FramePointerKindAttr)
    no_inline = opt_prop_def(UnitAttr)
    no_unwind = opt_prop_def(UnitAttr)
    optimize_none = opt_prop_def(UnitAttr)
    passthrough = opt_prop_def(ArrayAttr[Attribute])
    target_cpu = opt_prop_def(StringAttr)
    target_features = opt_prop_def(TargetFeaturesAttr)
    tune_cpu = opt_prop_def(StringAttr)
    unnamed_addr = opt_prop_def(IntegerAttr)

    def __init__(
        self,
        sym_name: str | StringAttr,
        function_type: LLVMFunctionType,
        linkage: LinkageAttr = LinkageAttr("internal"),
        cconv: CallingConventionAttr = CallingConventionAttr("ccc"),
        visibility: int | IntegerAttr[IntegerType] = 0,
        sym_visibility: str | StringAttr | None = None,
        body: Region | None = None,
        other_props: dict[str, Attribute | None] | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(visibility, int):
            visibility = IntegerAttr.from_int_and_width(visibility, 64)
        if body is None:
            body = Region([])
        if isinstance(sym_visibility, str):
            sym_visibility = StringAttr(sym_visibility)
        properties = other_props if other_props is not None else {}
        properties.update(
            {
                "sym_name": sym_name,
                "function_type": function_type,
                "CConv": cconv,
                "linkage": linkage,
                "visibility_": visibility,
                "sym_visibility": sym_visibility,
            }
        )
        super().__init__(
            operands=[],
            regions=[body],
            properties=properties,
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmreturn-mlirllvmreturnop).
    """

    name = "llvm.return"

    arg = opt_operand_def(Attribute)

    traits = traits_def(IsTerminator(), NoMemoryEffect())

    def __init__(self, value: Attribute | None = None):
        super().__init__(attributes={"value": value})


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "llvm.mlir.constant"
    result = result_def(Attribute)
    value = prop_def()

    traits = traits_def(NoMemoryEffect())

    def __init__(self, value: Attribute, value_type: Attribute):
        super().__init__(properties={"value": value}, result_types=[value_type])

    @classmethod
    def parse_value(cls, parser: Parser) -> Attribute:
        b = parser.parse_optional_boolean()
        if b is not None:
            return IntegerAttr.from_bool(b)
        attr = parser.parse_optional_attribute()
        if attr:
            return attr
        return IntegerAttr(parser.parse_integer(), 64)

    @classmethod
    def parse(cls, parser: Parser):
        parser.parse_characters("(")
        value = cls.parse_value(parser)
        parser.parse_characters(")")
        parser.parse_characters(":")
        value_type = parser.parse_type()
        return cls(value, value_type)

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            if isa(self.value, IntegerAttr) and self.result.type == IntegerType(64):
                self.value.print_without_type(printer)
            else:
                printer.print_attribute(self.value)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)


@irdl_attr_definition(init=False)
class FastMathAttr(FastMathAttrBase):
    name = "llvm.fastmath"


@irdl_op_definition
class CallIntrinsicOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcall_intrinsic-mlirllvmcallintrinsicop).
    """

    name = "llvm.call_intrinsic"

    fastmathFlags = opt_prop_def(FastMathAttr)
    intrin = prop_def(StringAttr)
    op_bundle_sizes = prop_def(DenseArrayBase.constr(i32))
    args = var_operand_def()
    op_bundle_operands = var_operand_def()
    ress = opt_result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        intrin: StringAttr | str,
        args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        *,
        op_bundle_sizes: DenseArrayBase,
        op_bundle_operands: Sequence[SSAValue] = (),
    ):
        if isinstance(intrin, str):
            intrin = StringAttr(intrin)
        super().__init__(
            operands=[args, op_bundle_operands],
            result_types=(result_types,),
            properties={
                "intrin": intrin,
                "op_bundle_sizes": op_bundle_sizes,
            },
        )


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "llvm.call"

    args = var_operand_def()
    op_bundle_operands = var_operand_def()

    callee = opt_prop_def(SymbolRefAttr)
    var_callee_type = opt_prop_def(LLVMFunctionType)
    fastmathFlags = prop_def(FastMathAttr, default_value=FastMathAttr("none"))
    CConv = prop_def(CallingConventionAttr, default_value=CallingConventionAttr("ccc"))
    op_bundle_sizes = prop_def(
        DenseArrayBase.constr(i32),
        default_value=DenseArrayBase.from_list(i32, ()),
    )
    TailCallKind = prop_def(
        TailCallKindAttr, default_value=TailCallKindAttr(TailCallKind.NONE)
    )
    returned = opt_result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        callee: str | SymbolRefAttr | StringAttr,
        *args: SSAValue | Operation,
        op_bundle_sizes: DenseArrayBase = DenseArrayBase.from_list(i32, ()),
        op_bundle_operands: tuple[SSAValue, ...] = (),
        return_type: Attribute | None = None,
        calling_convention: CallingConventionAttr = CallingConventionAttr("ccc"),
        fastmath: FastMathAttr = FastMathAttr(None),
        variadic_args: int = 0,
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        op_result_type = [return_type]
        if return_type is None:
            return_type = LLVMVoidType()
        input_types = [
            SSAValue.get(arg).type for arg in args[: len(args) - variadic_args]
        ]
        if variadic_args:
            var_callee_type = LLVMFunctionType(
                input_types, return_type, bool(variadic_args)
            )
        else:
            var_callee_type = None
        super().__init__(
            operands=[args, op_bundle_operands],
            properties={
                "callee": callee,
                "var_callee_type": var_callee_type,
                "fastmathFlags": fastmath,
                "CConv": calling_convention,
                "op_bundle_sizes": op_bundle_sizes,
            },
            result_types=op_result_type,
        )


LLVMType = (
    LLVMStructType | LLVMPointerType | LLVMArrayType | LLVMVoidType | LLVMFunctionType
)
LLVMTypeConstr = (
    base(LLVMStructType)
    | base(LLVMPointerType)
    | base(LLVMArrayType)
    | base(LLVMVoidType)
    | base(LLVMFunctionType)
)


@irdl_op_definition
class ZeroOp(IRDLOperation):
    name = "llvm.mlir.zero"

    assembly_format = "attr-dict `:` type($res)"

    traits = traits_def(NoMemoryEffect())

    res = result_def(LLVMTypeConstr)


class GenericCastOp(IRDLOperation, ABC):
    arg = operand_def(Attribute)
    """
    LLVM-compatible non-aggregate type
    """

    result = result_def(Attribute)
    """
    LLVM-compatible non-aggregate type
    """

    traits = traits_def(NoMemoryEffect())

    assembly_format = "$arg attr-dict `:` type($arg) `to` type($result)"

    def __init__(self, val: Operation | SSAValue, res_type: Attribute):
        super().__init__(
            operands=[SSAValue.get(val)],
            result_types=[res_type],
        )


class AbstractFloatArithOp(IRDLOperation, ABC):
    T: ClassVar = VarConstraint("T", AnyFloatConstr | VectorType.constr(AnyFloatConstr))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)

    fastmathFlags = prop_def(FastMathAttr, default_value=FastMathAttr(None))

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"

    irdl_options = [ParsePropInAttrDict()]

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        fast_math: FastMathAttr | FastMathFlag | None = None,
        attrs: dict[str, Attribute] | None = None,
    ):
        if isinstance(fast_math, FastMathFlag | str | None):
            fast_math = FastMathAttr(fast_math)

        super().__init__(
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
            properties={"fastmathFlags": fast_math},
            attributes=attrs,
        )


@irdl_op_definition
class FAddOp(AbstractFloatArithOp):
    name = "llvm.fadd"


@irdl_op_definition
class FMulOp(AbstractFloatArithOp):
    name = "llvm.fmul"


@irdl_op_definition
class FDivOp(AbstractFloatArithOp):
    name = "llvm.fdiv"


@irdl_op_definition
class FSubOp(AbstractFloatArithOp):
    name = "llvm.fsub"


@irdl_op_definition
class FRemOp(AbstractFloatArithOp):
    name = "llvm.frem"


@irdl_op_definition
class BitcastOp(GenericCastOp):
    name = "llvm.bitcast"


@irdl_op_definition
class SIToFPOp(GenericCastOp):
    name = "llvm.sitofp"


@irdl_op_definition
class FPExtOp(GenericCastOp):
    name = "llvm.fpext"


@irdl_op_definition
class UnreachableOp(IRDLOperation):
    name = "llvm.unreachable"

    traits = traits_def(IsTerminator())
    assembly_format = "attr-dict"


LLVM = Dialect(
    "llvm",
    [
        AShrOp,
        AddOp,
        AddressOfOp,
        AllocaOp,
        AndOp,
        BitcastOp,
        CallIntrinsicOp,
        CallOp,
        ConstantOp,
        ExtractValueOp,
        FAddOp,
        FDivOp,
        FMulOp,
        FPExtOp,
        FRemOp,
        FSubOp,
        FuncOp,
        GEPOp,
        GlobalOp,
        ICmpOp,
        InlineAsmOp,
        InsertValueOp,
        IntToPtrOp,
        LShrOp,
        LoadOp,
        MulOp,
        NullOp,
        OrOp,
        ReturnOp,
        SDivOp,
        SExtOp,
        SIToFPOp,
        SRemOp,
        ShlOp,
        StoreOp,
        SubOp,
        TruncOp,
        UDivOp,
        URemOp,
        UndefOp,
        UnreachableOp,
        XOrOp,
        ZExtOp,
        ZeroOp,
    ],
    [
        CallingConventionAttr,
        FastMathAttr,
        FramePointerKindAttr,
        LLVMArrayType,
        LLVMFunctionType,
        LLVMPointerType,
        LLVMStructType,
        LLVMVoidType,
        LinkageAttr,
        OverflowAttr,
        TailCallKindAttr,
        TargetFeaturesAttr,
    ],
)
