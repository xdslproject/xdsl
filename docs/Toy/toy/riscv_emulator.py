from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

from riscemu.config import RunConfig
from riscemu.instructions import (
    RV32D,
    RV32F,
    RV32I,
    RV32M,
    RV_Debug,
    Zicsr,
)
from riscemu.riscemu_main import RiscemuMain, RiscemuSource


def emulate_riscv(code: str) -> str:
    """
    Emulates RISC-V assembly using the default options for Toy tutorial notebooks.
    """
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=0,
        debug_on_exception=False,
        unlimited_registers=True,
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
    ]
    main.register_all_program_loaders()

    source = RiscemuSource("example.asm", StringIO(code))
    main.input_files.append(source)

    io = StringIO()
    with redirect_stdout(io):
        try:
            main.run()
        except Exception as ex:
            print(ex)
    return io.getvalue()
