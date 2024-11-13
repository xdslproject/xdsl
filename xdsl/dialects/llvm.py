from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import ClassVar

from xdsl.dialects.builtin import (
    I64,
    AnyIntegerAttr,
    ArrayAttr,
    ContainerType,
    DenseArrayBase,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i1,
    i32,
    i64,
)
from xdsl.dialects.utils import FastMathAttrBase
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
    BaseAttr,
    IRDLOperation,
    ParameterDef,
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
from xdsl.traits import IsTerminator, NoMemoryEffect, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr
from xdsl.utils.str_enum import StrEnum

GEP_USE_SSA_VAL = -2147483648
"""

This is used in the getelementptr index list to signify that an ssa value
should be used for this index.

"""


def _parse_llvm_type(parser: AttrParser):
    if parser.parse_optional_characters("void"):
        return LLVMVoidType()
    if parser.parse_optional_characters("ptr"):
        return LLVMPointerType(LLVMPointerType.parse_parameters(parser))
    if parser.parse_optional_characters("array"):
        return LLVMArrayType(LLVMArrayType.parse_parameters(parser))
    if parser.parse_optional_characters("struct"):
        return LLVMStructType(LLVMStructType.parse_parameters(parser))


def parse_llvm_type(parser: AttrParser):
    if (l := _parse_llvm_type(parser)) is not None:
        return l
    return parser.parse_attribute()


def parse_optional_llvm_type(parser: AttrParser):
    if (l := _parse_llvm_type(parser)) is not None:
        return l
    return parser.parse_optional_attribute()


@irdl_attr_definition
class LLVMStructType(ParametrizedAttribute, TypeAttribute):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types
    """

    name = "llvm.struct"

    # An empty string refers to a struct without a name.
    struct_name: ParameterDef[StringAttr]
    types: ParameterDef[ArrayAttr[Attribute]]

    # TODO: Add this parameter once xDSL supports the necessary capabilities.
    #  bitmask = ParameterDef(StringAttr)

    @staticmethod
    def from_type_list(types: Sequence[Attribute]) -> LLVMStructType:
        return LLVMStructType([StringAttr(""), ArrayAttr(types)])

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<")
        if self.struct_name.data:
            printer.print_string_literal(self.struct_name.data)
            printer.print_string(", ")
        printer.print("(")
        printer.print_list(self.types.data, printer.print_attribute)
        printer.print(")>")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
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
        return [StringAttr(struct_name), ArrayAttr(params)]


@irdl_attr_definition
class LLVMPointerType(
    ParametrizedAttribute, TypeAttribute, ContainerType[Attribute | None]
):
    name = "llvm.ptr"

    type: ParameterDef[Attribute | NoneAttr]
    addr_space: ParameterDef[IntAttr | NoneAttr]

    def print_parameters(self, printer: Printer) -> None:
        if isinstance(self.type, NoneAttr):
            return

        printer.print_string("<")
        printer.print_attribute(self.type)
        if not isinstance(self.addr_space, NoneAttr):
            printer.print_string(", ")
            printer.print_attribute(self.addr_space)

        printer.print_string(">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_characters("<") is None:
            return [NoneAttr(), NoneAttr()]
        type = parse_optional_llvm_type(parser)
        if type is None:
            parser.raise_error("Expected first parameter of llvm.ptr to be a type!")
        if parser.parse_optional_characters(",") is None:
            parser.parse_characters(">", " for llvm.ptr parameters")
            return [type, NoneAttr()]
        parser.parse_characters(",", " between llvm.ptr args")
        addr_space = parser.parse_integer()
        parser.parse_characters(">", " to end llvm.ptr parameters")
        return [type, IntegerAttr(addr_space, IndexType())]

    @staticmethod
    def opaque():
        return LLVMPointerType([NoneAttr(), NoneAttr()])

    @staticmethod
    def typed(type: Attribute):
        return LLVMPointerType([type, NoneAttr()])

    def is_typed(self):
        return not isinstance(self.type, NoneAttr)

    def get_element_type(self) -> Attribute | None:
        return self.type


@irdl_attr_definition
class LLVMArrayType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.array"

    size: ParameterDef[IntAttr]
    type: ParameterDef[Attribute]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_string(str(self.size.data))
        printer.print_string(" x ")
        printer.print_attribute(self.type)
        printer.print_string(">")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_characters("<") is None:
            return [NoneAttr(), NoneAttr()]
        size = IntAttr(parser.parse_integer())
        if parser.parse_optional_characters(">") is not None:
            return [size, NoneAttr()]
        parser.parse_shape_delimiter()
        type = parse_optional_llvm_type(parser)
        if type is None:
            parser.raise_error("Expected second parameter of llvm.array to be a type!")
        parser.parse_characters(">", " to end llvm.array parameters")
        return [size, type]

    @staticmethod
    def from_size_and_type(size: int | IntAttr, type: Attribute):
        if isinstance(size, int):
            size = IntAttr(size)
        return LLVMArrayType([size, type])


@irdl_attr_definition
class LLVMVoidType(ParametrizedAttribute, TypeAttribute):
    name = "llvm.void"


@irdl_attr_definition
class LLVMFunctionType(ParametrizedAttribute, TypeAttribute):
    """
    Currently does not support variadics.

    https://mlir.llvm.org/docs/Dialects/LLVM/#function-types
    """

    name = "llvm.func"

    inputs: ParameterDef[ArrayAttr[Attribute]]
    output: ParameterDef[Attribute]
    variadic: ParameterDef[UnitAttr | NoneAttr]

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
        super().__init__([inputs, output, variad_attr])

    @property
    def is_variadic(self) -> bool:
        return isinstance(self.variadic, UnitAttr)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        if isinstance(self.output, LLVMVoidType):
            printer.print("void")
        else:
            printer.print_attribute(self.output)

        printer.print(" (")
        printer.print_list(self.inputs, printer.print_attribute)
        if self.is_variadic:
            if self.inputs:
                printer.print(", ")
            printer.print("...")

        printer.print_string(")>")

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

    linkage: ParameterDef[StringAttr]

    def __init__(self, linkage: str | StringAttr) -> None:
        if isinstance(linkage, str):
            linkage = StringAttr(linkage)
        super().__init__([linkage])

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

    T: ClassVar = VarConstraint("T", BaseAttr(IntegerType))

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
        printer.print(" ", self.lhs, ", ", self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.lhs.type)


class OverflowFlag(StrEnum):
    NO_SIGNED_WRAP = "nsw"
    NO_UNSIGNED_WRAP = "nuw"


@dataclass(frozen=True, init=False)
class OverflowAttrBase(BitEnumAttribute[OverflowFlag]):
    none_value = "none"


@irdl_attr_definition
class OverflowAttr(OverflowAttrBase):
    name = "llvm.overflow"

    @classmethod
    def parse(cls, parser: Parser) -> OverflowAttr:
        if parser.parse_optional_keyword("overflow") is not None:
            return OverflowAttr(OverflowAttr.parse_parameter(parser))
        return OverflowAttr("none")

    def print(self, printer: Printer):
        if self.flags:
            printer.print(" overflow")
            self.print_parameter(printer)


class ArithmeticBinOpOverflow(IRDLOperation, ABC):
    """Class for arithmetic binary operations that use overflow flags."""

    T: ClassVar = VarConstraint("T", BaseAttr(IntegerType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)
    overflowFlags = opt_prop_def(OverflowAttr)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
        overflow: OverflowAttr = OverflowAttr(None),
    ):
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
        printer.print(" ", self.lhs, ", ", self.rhs)
        if self.overflowFlags:
            self.overflowFlags.print(printer)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.lhs.type)


class ArithmeticBinOpExact(IRDLOperation, ABC):
    """Class for arithmetic binary operations that use an exact flag."""

    T: ClassVar = VarConstraint("T", BaseAttr(IntegerType))

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
            printer.print(" exact")

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
        printer.print(" ", self.lhs, ", ", self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.lhs.type)


class ArithmeticBinOpDisjoint(IRDLOperation, ABC):
    """Class for arithmetic binary operations that use a disjoint flag."""

    T: ClassVar = VarConstraint("T", BaseAttr(IntegerType))

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
    arg = operand_def(IntegerType)

    res = result_def(IntegerType)

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
        printer.print(" ", self.arg)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.arg.type, " to ", self.res.type)


class IntegerConversionOpNNeg(IRDLOperation, ABC):
    arg = operand_def(IntegerType)
    res = result_def(IntegerType)
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
    arg = operand_def(IntegerType)
    res = result_def(IntegerType)
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
        printer.print(" ", self.arg)
        if self.overflowFlags:
            self.overflowFlags.print(printer)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.arg.type, " to ", self.res.type)


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
        assert isinstance(self.arg.type, IntegerType)
        assert isinstance(self.res.type, IntegerType)
        if self.arg.type.bitwidth <= self.res.type.bitwidth:
            raise VerifyException(
                f"invalid cast opcode for cast from {self.arg.type} to {self.res.type}"
            )
        super().verify(verify_nested_ops)


@irdl_op_definition
class ZExtOp(IntegerConversionOpNNeg):
    name = "llvm.zext"

    def verify(self, verify_nested_ops: bool = True):
        assert isinstance(self.arg.type, IntegerType)
        assert isinstance(self.res.type, IntegerType)
        if self.arg.type.bitwidth >= self.res.type.bitwidth:
            raise VerifyException(
                f"invalid cast opcode for cast from {self.arg.type} to {self.res.type}"
            )
        super().verify(verify_nested_ops)


@irdl_op_definition
class SExtOp(IntegerConversionOp):
    name = "llvm.sext"

    def verify(self, verify_nested_ops: bool = True):
        assert isinstance(self.arg.type, IntegerType)
        assert isinstance(self.res.type, IntegerType)
        if self.arg.type.bitwidth >= self.res.type.bitwidth:
            raise VerifyException(
                f"invalid cast opcode for cast from {self.arg.type} to {self.res.type}"
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
    T: ClassVar = VarConstraint("T", BaseAttr(IntegerType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(i1)
    predicate = prop_def(IntegerAttr[i64])

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        predicate: IntegerAttr[IntegerType],
        attributes: dict[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[i1],
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
        printer.print('" ', self.lhs, ", ", self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print(self.lhs.type)


@irdl_op_definition
class GEPOp(IRDLOperation):
    """
    llvm.getelementptr is an instruction to do pointer arithmetic by
    adding/subtracting offsets from a pointer.

    The llvm.getelementptr is documented in various places online:

    LLVM documentation: https://www.llvm.org/docs/GetElementPtr.html
    A good blogpost: https://blog.yossarian.net/2020/09/19/LLVMs-getelementptr-by-example
    MLIR documentation: https://mlir.llvm.org/docs/Dialects/LLVM/#llvmgetelementptr-mlirllvmgepop

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
    elem_type = opt_prop_def(Attribute)

    result = result_def(LLVMPointerType)

    rawConstantIndices = prop_def(DenseArrayBase)
    inbounds = opt_prop_def(UnitAttr)

    traits = traits_def(NoMemoryEffect())

    def __init__(
        self,
        ptr: SSAValue | Operation,
        indices: Sequence[int],
        ssa_indices: Sequence[SSAValue | Operation] | None = None,
        result_type: LLVMPointerType = LLVMPointerType.opaque(),
        inbounds: bool = False,
        pointee_type: Attribute | None = None,
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

        # convert a potential Operation into an SSAValue
        ptr_val = SSAValue.get(ptr)
        ptr_type = ptr_val.type

        if not isinstance(ptr_type, LLVMPointerType):
            raise ValueError("Input must be a pointer")

        props: dict[str, Attribute] = {
            "rawConstantIndices": DenseArrayBase.create_dense_int(i32, indices),
        }

        if not ptr_type.is_typed():
            if pointee_type is None:
                raise ValueError("Opaque types must have a pointee type passed")
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
        result_type: LLVMPointerType = LLVMPointerType.opaque(),
        inbounds: bool = False,
        pointee_type: Attribute | None = None,
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
            ssa_indices,
            result_type=result_type,
            inbounds=inbounds,
            pointee_type=pointee_type,
        )


@irdl_op_definition
class AllocaOp(IRDLOperation):
    name = "llvm.alloca"

    size = operand_def(IntegerType)

    alignment = opt_prop_def(AnyIntegerAttr)
    elem_type = opt_prop_def(Attribute)

    res = result_def()

    def __init__(
        self,
        size: SSAValue | Operation,
        elem_type: Attribute,
        alignment: int = 32,
        as_untyped_ptr: bool = True,
    ):
        props: dict[str, Attribute] = {
            "alignment": IntegerAttr.from_int_and_width(alignment, 64)
        }
        if as_untyped_ptr:
            ptr_type = LLVMPointerType.opaque()
            props["elem_type"] = elem_type
        else:
            ptr_type = LLVMPointerType.typed(elem_type)

        super().__init__(operands=[size], properties=props, result_types=[ptr_type])


@irdl_op_definition
class IntToPtrOp(IRDLOperation):
    name = "llvm.inttoptr"

    input = operand_def(IntegerType)

    output = result_def(LLVMPointerType)

    traits = traits_def(NoMemoryEffect())

    def __init__(self, input: SSAValue | Operation, ptr_type: Attribute | None = None):
        if ptr_type is None:
            ptr_type = LLVMPointerType.opaque()
        else:
            ptr_type = LLVMPointerType.typed(ptr_type)
        super().__init__(operands=[input], result_types=[ptr_type])


@irdl_op_definition
class InlineAsmOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvminline_asm-llvminlineasmop

    To see what each field means, have a look at:
    https://llvm.org/docs/LangRef.html#inline-assembler-expressions
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

    def __init__(
        self,
        asm_string: str,
        constraints: str,
        operands: Sequence[SSAValue | Operation],
        res_types: Sequence[Attribute] | None = None,
        asm_dialect: int = 0,
        has_side_effects: bool = False,
        is_align_stack: bool = False,
    ):
        props: dict[str, Attribute | None] = {
            "asm_string": StringAttr(asm_string),
            "constraints": StringAttr(constraints),
            "asm_dialect": IntegerAttr.from_int_and_width(asm_dialect, 64),
            "has_side_effects": UnitAttr() if has_side_effects else None,
            "is_align_stack": UnitAttr() if is_align_stack else None,
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

    ordering = opt_prop_def(IntegerAttr[IntegerType])

    dereferenced_value = result_def()

    def __init__(self, ptr: SSAValue | Operation, result_type: Attribute | None = None):
        if result_type is None:
            ptr = SSAValue.get(ptr)
            assert isinstance(ptr.type, LLVMPointerType)

            if isinstance(ptr.type.type, NoneAttr):
                raise ValueError(
                    "llvm.load requires either a result type or a typed pointer!"
                )
            result_type = ptr.type.type

        super().__init__(operands=[ptr], result_types=[result_type])


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
            ptr_type = LLVMPointerType.opaque()
        assert isinstance(ptr_type, LLVMPointerType)

        super().__init__(result_types=[ptr_type])


@irdl_op_definition
class ExtractValueOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmextractvalue-mlirllvmextractvalueop
    """

    name = "llvm.extractvalue"

    position = prop_def(DenseArrayBase)
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
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvminsertvalue-mlirllvminsertvalueop
    """

    name = "llvm.insertvalue"

    position = prop_def(DenseArrayBase)
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
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirundef-mlirllvmundefop
    """

    name = "llvm.mlir.undef"

    res = result_def(Attribute)

    traits = traits_def(NoMemoryEffect())

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])


@irdl_op_definition
class GlobalOp(IRDLOperation):
    name = "llvm.mlir.global"

    global_type = prop_def(Attribute)
    constant = opt_prop_def(UnitAttr)
    sym_name = prop_def(StringAttr)
    linkage = prop_def(LinkageAttr)
    dso_local = opt_prop_def(UnitAttr)
    thread_local_ = opt_prop_def(UnitAttr)
    visibility_ = opt_prop_def(IntegerAttr[IntegerType])
    value = opt_prop_def(Attribute)
    alignment = opt_prop_def(AnyIntegerAttr)
    addr_space = prop_def(AnyIntegerAttr)
    unnamed_addr = opt_prop_def(AnyIntegerAttr)
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

        super().__init__(properties=props, regions=[Region([])])


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
A list of all valid calling conventions understood by LLVM, see
https://llvm.org/docs/LangRef.html#calling-conventions
for more info.
"""


@irdl_attr_definition
class CallingConventionAttr(ParametrizedAttribute):
    """
    LLVM Calling convention, default is ccc.
    """

    name = "llvm.cconv"

    convention: ParameterDef[StringAttr]

    @property
    def cconv_name(self) -> str:
        return self.convention.data

    def __init__(self, conv: str):
        super().__init__([StringAttr(conv)])

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


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "llvm.func"

    body = region_def()
    sym_name = prop_def(StringAttr)
    function_type = prop_def(LLVMFunctionType)
    CConv = prop_def(CallingConventionAttr)
    linkage = prop_def(LinkageAttr)
    sym_visibility = opt_prop_def(StringAttr)
    visibility_ = prop_def(IntegerAttr[IntegerType])

    def __init__(
        self,
        sym_name: str | StringAttr,
        function_type: LLVMFunctionType,
        linkage: LinkageAttr = LinkageAttr("internal"),
        cconv: CallingConventionAttr = CallingConventionAttr("ccc"),
        visibility: int | IntegerAttr[IntegerType] = 0,
        sym_visibility: str | StringAttr | None = None,
        body: Region | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(visibility, int):
            visibility = IntegerAttr.from_int_and_width(visibility, 64)
        if body is None:
            body = Region([])
        if isinstance(sym_visibility, str):
            sym_visibility = StringAttr(sym_visibility)
        super().__init__(
            operands=[],
            regions=[body],
            properties={
                "sym_name": sym_name,
                "function_type": function_type,
                "CConv": cconv,
                "linkage": linkage,
                "visibility_": visibility,
                "sym_visibility": sym_visibility,
            },
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmreturn-mlirllvmreturnop
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
    value = prop_def(Attribute)

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
        printer.print("(")
        if isattr(self.value, AnyIntegerAttr) and self.result.type == IntegerType(64):
            self.value.print_without_type(printer)
        else:
            printer.print(self.value)
        printer.print(") : ", self.result.type)


@irdl_attr_definition
class FastMathAttr(FastMathAttrBase):
    name = "llvm.fastmath"


@irdl_op_definition
class CallIntrinsicOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcall_intrinsic-mlirllvmcallintrinsicop
    """

    name = "llvm.call_intrinsic"

    fastmathFlags = opt_prop_def(FastMathAttr)
    intrin = prop_def(StringAttr)
    args = var_operand_def()
    ress = opt_result_def()

    def __init__(
        self,
        intrin: StringAttr | str,
        args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
    ):
        if isinstance(intrin, str):
            intrin = StringAttr(intrin)
        super().__init__(
            operands=args,
            result_types=(result_types,),
            properties={
                "intrin": intrin,
            },
        )


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
        printer.print_string("<")
        super().print_parameter(printer)
        printer.print_string(">")


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "llvm.call"

    args = var_operand_def()

    callee = opt_prop_def(SymbolRefAttr)
    var_callee_type = opt_prop_def(LLVMFunctionType)
    fastmathFlags = prop_def(FastMathAttr, default_value=FastMathAttr("none"))
    CConv = prop_def(CallingConventionAttr, default_value=CallingConventionAttr("ccc"))
    TailCallKind = prop_def(
        TailCallKindAttr, default_value=TailCallKindAttr(TailCallKind.NONE)
    )
    returned = opt_result_def()

    def __init__(
        self,
        callee: str | SymbolRefAttr | StringAttr,
        *args: SSAValue | Operation,
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
        var_callee_type = LLVMFunctionType(input_types, return_type, variadic_args > 0)
        super().__init__(
            operands=[args],
            properties={
                "callee": callee,
                "var_callee_type": var_callee_type,
                "fastmathFlags": fastmath,
                "CConv": calling_convention,
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


LLVM = Dialect(
    "llvm",
    [
        AddOp,
        SubOp,
        MulOp,
        UDivOp,
        SDivOp,
        URemOp,
        SRemOp,
        AndOp,
        OrOp,
        XOrOp,
        ShlOp,
        LShrOp,
        AShrOp,
        TruncOp,
        ZExtOp,
        SExtOp,
        ICmpOp,
        ExtractValueOp,
        InsertValueOp,
        InlineAsmOp,
        UndefOp,
        AllocaOp,
        GEPOp,
        IntToPtrOp,
        NullOp,
        LoadOp,
        StoreOp,
        GlobalOp,
        AddressOfOp,
        FuncOp,
        CallOp,
        ReturnOp,
        ConstantOp,
        CallIntrinsicOp,
        ZeroOp,
    ],
    [
        LLVMStructType,
        LLVMPointerType,
        LLVMArrayType,
        LLVMVoidType,
        LLVMFunctionType,
        LinkageAttr,
        CallingConventionAttr,
        TailCallKindAttr,
        FastMathAttr,
        OverflowAttr,
    ],
)
