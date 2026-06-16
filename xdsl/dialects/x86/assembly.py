from __future__ import annotations

from typing import TypeAlias

from xdsl.backend.assembly_printer import reg
from xdsl.dialects.builtin import IntegerAttr, StringAttr, UnitAttr
from xdsl.ir import SSAValue
from xdsl.parser import Parser
from xdsl.printer import Printer

AssemblyInstructionArg: TypeAlias = IntegerAttr | str | StringAttr


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, IntegerAttr):
        return f"{arg.value.data}"
    elif isinstance(arg, StringAttr):
        return arg.data

    return arg


def memory_access_str(register: SSAValue, offset: IntegerAttr) -> str:
    register_str = reg(register)
    if offset.value.data:
        mem_acc_str = f"[{register_str}{offset.value.data:+d}]"
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


def masked_memory_access_str(
    register: SSAValue,
    offset: IntegerAttr,
    mask: SSAValue,
    z: UnitAttr | None,
) -> str:
    """
    Returns string for asm printing of a memory access followed by the {k}
    (and optionally {z}) specifiers, in AVX512 masked operations.
    e.g. ``[rdx+8] {k1}`` or ``[rdx] {k1}{z}``
    """
    mem_str = memory_access_str(register, offset)
    mask_str = reg(mask)
    res = f"{mem_str} {{{mask_str}}}"
    if z:
        res += "{z}"
    return res


def masked_source_str(reg_in: SSAValue, mask: SSAValue, z: UnitAttr | None) -> str:
    """
    Returns string for asm printing of the register followed by the {k} (and optionally {z})
    specifiers, in AVX512 masked operations
    """
    reg_in_str = reg(reg_in)
    mask_str = reg(mask)
    res = f"{reg_in_str} {{{mask_str}}}"
    if z:
        res += "{z}"
    return res
