from __future__ import annotations

from typing import TypeAlias

from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, UnitAttr
from xdsl.ir import SSAValue
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.hints import isa

from .attributes import LabelAttr
from .registers import (
    AVX512MaskRegisterType,
    GeneralRegisterType,
    RFLAGSRegisterType,
    X86VectorRegisterType,
)

AssemblyInstructionArg: TypeAlias = (
    IntegerAttr | SSAValue | GeneralRegisterType | str | int | LabelAttr
)


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isa(arg, IntegerAttr):
        return f"{arg.value.data}"
    elif isinstance(arg, int):
        return f"{arg}"
    elif isinstance(arg, str):
        return arg
    elif isinstance(arg, GeneralRegisterType):
        return arg.register_name.data
    elif isinstance(arg, RFLAGSRegisterType):
        return arg.register_name.data
    elif isinstance(arg, X86VectorRegisterType):
        return arg.register_name.data
    elif isinstance(arg, LabelAttr):
        return arg.data
    else:
        if isinstance(arg.type, GeneralRegisterType):
            reg = arg.type.register_name
            return reg.data
        elif isinstance(arg.type, RFLAGSRegisterType):
            reg = arg.type.register_name
            return reg.data
        elif isinstance(arg.type, X86VectorRegisterType):
            reg = arg.type.register_name
            return reg.data
        elif isinstance(arg.type, AVX512MaskRegisterType):
            reg = arg.type.register_name
            return f"{{{reg.data}}}"
        else:
            raise ValueError(f"Unexpected register type {arg.type}")


def parse_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr:
    return parser.expect(
        lambda: parse_optional_immediate_value(parser, integer_type),
        "Expected immediate",
    )


def parse_optional_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr | None:
    """
    Parse an optional immediate value. If an integer is parsed, an integer attr with the specified type is created.
    """
    if (immediate := parser.parse_optional_integer()) is not None:
        return IntegerAttr(immediate, integer_type)
    if (immediate := parser.parse_optional_str_literal()) is not None:
        return LabelAttr(immediate)


def print_immediate_value(printer: Printer, immediate: IntegerAttr | LabelAttr):
    match immediate:
        case IntegerAttr():
            immediate.print_without_type(printer)
        case LabelAttr():
            printer.print_string_literal(immediate.data)


def memory_access_str(register: AssemblyInstructionArg, offset: IntegerAttr) -> str:
    register_str = assembly_arg_str(register)
    if offset.value.data != 0:
        offset_str = assembly_arg_str(offset)
        if offset.value.data > 0:
            mem_acc_str = f"[{register_str}+{offset_str}]"
        else:
            mem_acc_str = f"[{register_str}{offset_str}]"
    else:
        mem_acc_str = f"[{register_str}]"
    return mem_acc_str


def print_type_pair(printer: Printer, value: SSAValue) -> None:
    printer.print_ssa_value(value)
    printer.print_string(" : ")
    printer.print_attribute(value.type)


def parse_type_pair(parser: Parser) -> SSAValue:
    unresolved = parser.parse_unresolved_operand()
    parser.parse_punctuation(":")
    type = parser.parse_type()
    return parser.resolve_operand(unresolved, type)


def masked_source_str(reg_in: SSAValue, mask: SSAValue, z: UnitAttr | None) -> str:
    """
    Returns string for asm printing of the register followed by the {k} (and optionally {z})
    specifiers, in AVX512 masked operations
    """
    register_in = assembly_arg_str(reg_in) + " " + assembly_arg_str(mask)
    if z:
        register_in += "{z}"
    return register_in
