"""
This file provides tools to describe toolchain targets.

Based on https://github.com/riscv-non-isa/riscv-elf-psabi-doc

And the RISC-V specification
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

_ISA_ORDER = "IEMAFDGQLCBJTPVN"
_ISA_EXT_ORDER = ("Zicsr", "Zifencei", "Zam", "Ztso")


def _isa_sort_key(ext: str) -> int:
    """
    Helper to correctly sort a RISC-V arch name.

    First come the letters from _ISA_ORDER
    Then the extensions from _ISA_EXT_ORDER
    And then custom extensions, prefixed with X
    """
    if ext in _ISA_ORDER:
        return _ISA_ORDER.index(ext)
    if ext[0] == "Z":
        if ext in _ISA_EXT_ORDER:
            return 100 + _ISA_EXT_ORDER.index(ext)
        return 190
    return 200


def _expand_isa_letters(extensions_: list[str]) -> list[str]:
    """
    Normalizes (expands) ISA extensions as per RISC-V ISA Manual version 20191213

    See Section 27.11 at:
    https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf
    """
    extensions = set(extensions_)

    if "G" in extensions:
        extensions.remove("G")
        extensions.update(("I", "M", "A", "D", "Zifencei"))
    if "D" in extensions:
        extensions.add("F")
    if "Q" in extensions:
        extensions.add("D")
    if "F" in extensions:
        extensions.add("Zicsr")
    if "Zam" in extensions:
        extensions.add("A")

    return list(sorted(extensions, key=_isa_sort_key))


@dataclass(frozen=True, unsafe_hash=True)
class ABISpec:
    """
    This defines the ABI.

    Based on
    https://github.com/riscv-non-isa/riscv-toolchain-conventions#specifying-the-target-abi-with--mabi
    (better source might be nice though)
    """

    # various type bitwidths:
    int_width: int
    long_width: int
    index_width: int

    # stack alignment:
    stack_alignment: int

    # argument passing:
    call_with_floats: Literal[None, 32, 64]
    """
    Are the floating point registers used to pass arguments?
    
    None => No
    32   => 32-bit fp registers are used
    64   => 64-bit fp registers are used
    """

    # binary file format:
    file_format: str = "elf"
    """
    The output file format (for example of object files).
    
    "elf" is the default, and I don't know of any others.
    """


@dataclass(frozen=True, unsafe_hash=True, init=False, repr=False)
class MachineArchSpec:
    """
    Machine architecture spec, bitwidth, extensions, etc.
    """

    xlen: int
    """
    Register size, basically 32/64, for RV32/RV64 respectively
    """
    flen: int
    """
    Floating point register width (32/64/128)
    """

    extensions: list[str]
    """
    A list of extensions, fully expanded. 
    
    RV32G would be: ["I", "M", "A", "F", "D", "Zifencei", "Zicsr", "Zam"]
    """

    def __repr__(self):
        return 'MachineArchSpec("RV{}{}")'.format(self.xlen, "".join(self.extensions))

    def __init__(self, march: str):
        # make it not case-sensitive
        march = march.upper()

        if not march.startswith("RV"):
            raise ValueError("Spec must start with RV...")

        match = re.fullmatch(r"(\d+)([A-Y]*)((Z[A-Y]+)*)((X[A-WYZ]+)*)", march[2:])
        if match is None:
            raise ValueError(f'Malformed march string: "{march}"')
        width_str, letters, exts, _, more_exts, _ = match.groups()

        # set bitwidth
        object.__setattr__(self, "xlen", int(width_str))

        # normalize extensions
        object.__setattr__(
            self,
            "extensions",
            _expand_isa_letters(
                list(letters)
                + ["Z" + z.lower() for z in exts.split("Z")[1:]]
                + ["X" + x.lower() for x in more_exts.split("X")[1:]]
            ),
        )

        # determine flen
        if "Q" in self.extensions:
            flen = 128
        elif "D" in self.extensions:
            flen = 64
        elif "F" in self.extensions:
            flen = 32
        else:
            flen = 0
        object.__setattr__(self, "flen", flen)


@dataclass
class TargetDefinition:
    abi: ABISpec
    """
    Target ABI (type bitwidth, stack alignment, argument passing)
    """

    march: MachineArchSpec
    """
    Machine architecture (handled by -march=RV...)
    """

    code_model: Literal["any", "low"] = "any"
    """
    Code model (usually handled by -mcmodel=med<model>):
    https://github.com/riscv-non-isa/riscv-toolchain-conventions#specifying-the-target-code-model-with--mcmodel
    
    low: The program and its statically defined symbols must lie within a single 2GiB 
    address range, between the absolute addresses -2GiB and +2GiB. lui and addi pairs 
    are used to generate addresses.
    
    any: The program and its statically defined symbols must lie within a single 4GiB 
    address range. auipc and addi pairs are used to generate addresses.
    """


class MAbi:
    """
    Collection of common -mabi values
    """

    ILP32 = ABISpec(32, 32, 32, stack_alignment=32, call_with_floats=None)
    ILP32F = ABISpec(32, 32, 32, stack_alignment=32, call_with_floats=32)
    ILP32D = ABISpec(32, 32, 32, stack_alignment=64, call_with_floats=64)

    LP32 = ABISpec(32, 64, 64, stack_alignment=64, call_with_floats=None)
    LP32F = ABISpec(32, 64, 64, stack_alignment=64, call_with_floats=32)
    LP32D = ABISpec(32, 64, 64, stack_alignment=64, call_with_floats=64)


class RecognizedTargets(Enum):
    riscv32_riscemu = TargetDefinition(MAbi.ILP32, MachineArchSpec("RV32IMAZto"))
    riscv64_linux = TargetDefinition(MAbi.ILP32D, MachineArchSpec("RV64G"))
