from __future__ import annotations
from typing import Optional

# pyright: reportMissingTypeStubs=false

from riscemu import RunConfig, UserModeCPU, RV32I, RV32M, AssemblyFileLoader, MMU
from riscemu.instructions import InstructionSet

from io import StringIO

def run_riscv(
    code: str,
    extensions: list[type[InstructionSet]] = [],
    unlimited_regs: bool = False,
    setup_stack: bool = False,
    verbosity: int = 5,
) -> Optional[int]:
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=verbosity,
        debug_on_exception=False,
        unlimited_registers=unlimited_regs,
    )

    cpu = UserModeCPU([RV32I, RV32M, *extensions], cfg)

    io = StringIO(code)

    loader = AssemblyFileLoader.instantiate("example.asm", {})
    assert isinstance(loader, AssemblyFileLoader)
    cpu.load_program(loader.parse_io(io))  # pyright: ignore[reportUnknownMemberType]

    mmu: MMU = getattr(cpu, "mmu")
    try:
        if setup_stack:
            cpu.setup_stack(cfg.stack_size)
        cpu.launch(mmu.programs[-1], verbosity > 1)
        return cpu.exit_code
    except Exception as ex:
        print(ex)
        return None
