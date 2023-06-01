from __future__ import annotations
from typing import IO, ClassVar

from riscemu.config import RunConfig
from riscemu import RV32I, RV32M
from riscemu.CPU import UserModeCPU
from riscemu.parser import AssemblyFileLoader
from riscemu.instructions.instruction_set import InstructionSet
from riscemu.types.instruction import Instruction

from io import StringIO


class RV_Debug(InstructionSet):
    stream: ClassVar[IO[str] | None] = None

    # riscemu matches `instruction_` prefixes, so this will be called by `print reg`
    def instruction_print(self, ins: Instruction):
        reg = ins.get_reg(0)
        value = self.regs.get(reg)
        print(value, file=type(self).stream)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, RV_Debug):
            return False
        return self.stream is __value.stream

    def __hash__(self) -> int:
        return hash(id(self.stream))


def run_riscv(
    code: str,
    extensions: list[type[InstructionSet]] = [],
    unlimited_regs: bool = False,
    verbosity: int = 5,
):
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=verbosity,
        debug_on_exception=False,
        unlimited_registers=unlimited_regs,
    )

    cpu = UserModeCPU([RV32I, RV32M, RV_Debug, *extensions], cfg)

    io = StringIO(code)

    loader = AssemblyFileLoader.instantiate("example.asm", {})
    assert isinstance(loader, AssemblyFileLoader)
    cpu.load_program(loader.parse_io(io))

    try:
        cpu.launch(verbosity > 1)
    except Exception as ex:
        print(ex)
