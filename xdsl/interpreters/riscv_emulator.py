from __future__ import annotations

from collections.abc import Sequence
from io import StringIO

from riscemu.config import RunConfig
from riscemu.instructions import (
    RV32D,
    RV32F,
    RV32I,
    RV32M,
    InstructionSet,
    RV_Debug,
    Zicsr,
)
from riscemu.riscemu_main import RiscemuMain, RiscemuSource


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
        flen=64,
    )

    main = RiscemuMain(cfg)
    main.selected_ins_sets = [
        RV32I,
        RV32M,
        RV32F,
        RV32D,
        Zicsr,
        RV_Debug,
        *extensions,
    ]
    main.register_all_program_loaders()

    source = RiscemuSource("example.asm", StringIO(code))
    main.input_files.append(source)

    try:
        main.run()
    except Exception as ex:
        print(ex)
