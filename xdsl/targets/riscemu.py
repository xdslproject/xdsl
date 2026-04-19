from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import IO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.target import Target


@dataclass(frozen=True)
class RISCVEmulatorTarget(Target):
    name = "riscemu"

    unlimited_regs: bool = True
    verbosity: int = 0

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        try:
            from xdsl.interpreters.riscv_emulator import run_riscv
        except ImportError:
            print("Please install optional dependencies to run riscv emulation")
            return

        from xdsl.dialects.riscv import riscv_code

        code = riscv_code(module)
        with redirect_stdout(output):
            run_riscv(
                code,
                unlimited_regs=self.unlimited_regs,
                verbosity=self.verbosity,
            )
