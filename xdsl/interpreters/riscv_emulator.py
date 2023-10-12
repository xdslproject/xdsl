from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
from typing import IO, ClassVar

from riscemu.config import RunConfig
from riscemu.core.instruction import Instruction
from riscemu.instructions.instruction_set import InstructionSet
from riscemu.instructions.RV32F import RV32F
from riscemu.instructions.RV32I import RV32I
from riscemu.instructions.RV32M import RV32M
from riscemu.instructions.Zicsr import Zicsr
from riscemu.riscemu_main import RiscemuMain, RiscemuSource


class RV_Debug(InstructionSet):
    stream: ClassVar[IO[str] | None] = None

    # riscemu matches `instruction_` prefixes, so this will be called by `print reg`

    def instruction_print(self, ins: Instruction):
        reg = ins.get_reg(0)
        value = self.regs.get(reg)
        print(value, file=type(self).stream)

    def instruction_print_float(self, ins: Instruction):
        reg = ins.get_reg(0)
        value = self.regs.get_f(reg).value
        print(value, file=type(self).stream)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, RV_Debug):
            return False
        return self.stream is __value.stream

    def __hash__(self) -> int:
        return hash(id(self.stream))


def run_riscv(
    code: str,
    extensions: Sequence[type[InstructionSet]] = (),
    unlimited_regs: bool = False,
    verbosity: int = 5,
):
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=verbosity,
        debug_on_exception=False,
        unlimited_registers=unlimited_regs,
        use_libc=True,
    )

    main = RiscemuMain(cfg)
    main.selected_ins_sets = [RV32I, RV32M, RV32F, Zicsr, RV_Debug, *extensions]
    main.register_all_program_loaders()

    source = RiscemuSource("example.asm", StringIO(code))
    main.input_files.append(source)

    try:
        main.run()
    except Exception as ex:
        print(ex)
