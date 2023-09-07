"""
This file provides tools to describe toolchain targets.

Based on:

 [0]: https://github.com/riscv-non-isa/riscv-elf-psabi-doc
 [1]: `RISC-V ABIs Specification, Document Version 1.0', Editors Kito Cheng and Jessica
      Clarke, RISC-V International, November 2022.
      https://github.com/riscv-non-isa/riscv-elf-psabi-doc/releases/tag/v1.0
 [2]: “The RISC-V Instruction Set Manual, Volume I: User-Level ISA, Document Version
      20191213”, Editors Andrew Waterman and Krste Asanovi´c, RISC-V Foundation,
      December 2019.
      https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf

"""

from __future__ import annotations

import re
from collections.abc import Sequence
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

    Implemented according to chapter 27 in [2].
    """
    if ext in _ISA_ORDER:
        return _ISA_ORDER.index(ext)
    if ext[0] == "Z":
        if ext in _ISA_EXT_ORDER:
            return 100 + _ISA_EXT_ORDER.index(ext)
        return 190
    return 200


def _expand_isa_letters(extensions_: Sequence[str]) -> tuple[str, ...]:
    """
    Normalizes (expands) ISA extensions as per RISC-V ISA Manual version 20191213

    See Section 27.11 in [2] for reference.
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

    return tuple(sorted(extensions, key=_isa_sort_key))


@dataclass(frozen=True, unsafe_hash=True)
class ABISpec:
    """
    This defines the ABI.

    Largely based on chapter 2 of [1]
    """

    # various type bitwidths:
    int_width: int
    long_width: int
    pointer_width: int

    # The ABIs flen is allowed to smaller than the machine flen
    abi_flen: Literal[0, 32, 64, 128]
    """
    ABI_FLEN refers to the width of a floating-point register in the ABI. The ABI_FLEN
    must be no wider than the ISA’s FLEN. The ISA might have wider floating-point
    registers than the ABI.

    (cited from section 2.2 [1], p. 10)
    """

    # binary file format:
    file_format: str = "elf"
    """
    The output file format (for example of object files).

    "elf" is the default for most
    """

    # stack alignment:
    stack_alignment: int = 128
    """
    The stack grows downwards (towards lower addresses) and the stack pointer shall be
    aligned to a 128-bit boundary upon procedure entry. The first argument passed on the
    stack is located at offset zero of the stack pointer on function entry; following
    arguments are stored at correspondingly higher addresses.

    (cited from section 2.1 [1] p. 9)

    A supporting illustration:

    +--------------------------+ <--- stack "end", higher address
    |       In-use stack       |
    | (from calling functions) |
    +--------------------------+
    |    Padding if needed     |
    +--------------------------+
    |     stack argument n     |
    |     stack argument n-1   |
    |            ...           |
    |     stack argument 0     |
    +--------------------------+ <--- Stack pointer, aligned to 128 bits
    |           empty          |
    |                          |
    +--------------------------+ <--- stack "start", lower address

    This means `0(sp)` is stack argument 0, and `4*n(sp)` is stack argument n.
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
    Floating point register width (0/32/64/128)
    """

    extensions: tuple[str, ...]
    """
    A list of extensions, fully expanded.

    RV32G would be: ["I", "M", "A", "F", "D", "Zifencei", "Zicsr"]
    """

    @property
    def spec_string(self) -> str:
        i = 0
        for i, e in enumerate(self.extensions):
            if len(e) > 1:
                break
        return "".join(self.extensions[:i]) + "_".join(self.extensions[i:])

    def __repr__(self):
        return f'MachineArchSpec("RV{self.xlen}{self.spec_string}")'

    def __init__(self, march: str):
        if not march.startswith("RV"):
            raise ValueError("Spec must start with RV...")

        match = re.fullmatch(
            r"RV(\d+)([A-Y]*)((Z[a-z]+)?(_Z[a-z]+)*)_?((X[a-z]+)?(_X[a-z]+)*)", march
        )
        if match is None:
            raise ValueError(f'Malformed march string: "{march}"')
        width_str, letters, exts, _, _, more_exts, _, _ = match.groups()

        # set bitwidth
        object.__setattr__(self, "xlen", int(width_str))

        # normalize extensions
        object.__setattr__(
            self,
            "extensions",
            _expand_isa_letters(
                list(letters)
                + ["Z" + z.lower().strip("_") for z in exts.split("Z")[1:]]
                + ["X" + x.lower().strip("_") for x in more_exts.split("X")[1:]]
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

    def supports_mabi(self, abi: ABISpec) -> bool:
        """
        Implements checks lined out in section 2.4 of [2], p. 12
        """
        # check that abi flen is not larger than the march flen
        if abi.abi_flen > self.flen:
            return False
        # check that ilp32* is only used on RV32*
        # and lp64* is only used on RV64*
        if abi.pointer_width != self.xlen:
            return False
        if abi.long_width != self.xlen:
            return False
        # check that an int always fits in a single register
        if abi.int_width > self.xlen:
            return False
        return True


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

    def is_valid(self) -> bool:
        return self.march.supports_mabi(self.abi) and self.code_model in ("any", "low")


class MAbi(Enum):
    """
    Collection of named ABIs as per chapter 2.4 of [1].

    ILP32E is omitted as the E extension is currently not ratified.
    """

    ILP32 = ABISpec(32, 32, 32, abi_flen=0)
    ILP32F = ABISpec(32, 32, 32, abi_flen=32)
    ILP32D = ABISpec(32, 32, 32, abi_flen=64)

    LP64 = ABISpec(32, 64, 64, abi_flen=0)
    LP64F = ABISpec(32, 64, 64, abi_flen=32)
    LP64D = ABISpec(32, 64, 64, abi_flen=64)
    LP64Q = ABISpec(32, 64, 64, abi_flen=128)


class RecognizedTargets(Enum):
    riscv32_riscemu = TargetDefinition(MAbi.ILP32.value, MachineArchSpec("RV32IMA_Zto"))
    riscv64_linux = TargetDefinition(MAbi.LP64D.value, MachineArchSpec("RV64G"))
    snitch = TargetDefinition(
        MAbi.ILP32D.value, MachineArchSpec("RV32IMAD_Xssr_Xfrep_Xdma")
    )
