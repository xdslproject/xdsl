import pytest

from xdsl.backend.register_stack import OutOfRegisters
from xdsl.backend.x86.arch import AVX2, AVX512, UNKNOWN, X86Arch
from xdsl.backend.x86.register_stack import X86RegisterStack
from xdsl.dialects import x86


def test_default_reserved_registers():
    register_stack = X86RegisterStack.get()

    for reg in (
        x86.registers.RAX,
        x86.registers.RDX,
        x86.registers.RSP,
    ):
        available_before = register_stack.available_registers.copy()
        register_stack.push(reg)
        assert available_before == register_stack.available_registers


def test_push_infinite_register():
    register_stack = X86RegisterStack(allow_infinite=True)

    infinite0 = x86.AVX2RegisterType.infinite_register(0)
    register_stack.push(infinite0)
    assert register_stack.pop(x86.AVX2RegisterType) == infinite0


def test_push_register():
    register_stack = X86RegisterStack.get()

    register_stack.push(x86.registers.YMM0)
    assert register_stack.pop(x86.AVX2RegisterType) == x86.registers.YMM0

    register_stack.push(x86.registers.RAX)
    assert register_stack.pop(x86.registers.Reg64Type) == x86.registers.RAX


def test_gpr_widths_share_one_pool():
    """
    64/32/16/8-bit GPR names share pool key x86.reg; exhausting one width blocks others.
    """
    stack = X86RegisterStack.get()

    try:
        while True:
            stack.pop(x86.registers.Reg64Type)
    except OutOfRegisters:
        pass

    with pytest.raises(OutOfRegisters):
        stack.pop(x86.registers.Reg32Type)


def test_vector_registers_share_one_pool():
    """
    All x86 vector kinds share one pool key; indices 0..15 alias across xmm/ymm/zmm.
    After every ABI index usable by SSE is taken, AVX2 cannot allocate (same indices).
    """
    stack = X86RegisterStack.get()

    try:
        while True:
            stack.pop(x86.registers.SSERegisterType)
    except OutOfRegisters:
        pass

    with pytest.raises(OutOfRegisters):
        stack.pop(x86.registers.AVX2RegisterType)


def test_default_stack_stops_at_the_vex_boundary():
    """
    Without a target the allocator must assume VEX, which reaches ymm0-15 only.

    ymm16-31 exist in the register file but can only be named through EVEX, so
    handing them out by default emits instructions the target may not decode.
    """
    register_stack = X86RegisterStack.get()

    for _ in range(16):
        register_stack.pop(x86.AVX2RegisterType)

    with pytest.raises(OutOfRegisters):
        register_stack.pop(x86.AVX2RegisterType)


@pytest.mark.parametrize(
    "arch, expected",
    [(UNKNOWN, 16), (AVX2, 16), (AVX512, 32)],
)
def test_vector_registers_available_per_arch(arch: X86Arch, expected: int):
    """AVX-512 is the only target that can name the upper half of the bank."""
    register_stack = X86RegisterStack.get_for_arch(arch)

    for _ in range(expected):
        register_stack.pop(x86.AVX2RegisterType)

    with pytest.raises(OutOfRegisters):
        register_stack.pop(x86.AVX2RegisterType)


@pytest.mark.parametrize("arch", [UNKNOWN, AVX2, AVX512])
def test_general_purpose_registers_do_not_vary_by_arch(arch: X86Arch):
    """Only the vector half of the set is target dependent."""
    assert [
        reg
        for reg in arch.default_allocatable_registers()
        if isinstance(reg, x86.registers.Reg64Type)
    ] == list(x86.registers.Reg64Type.allocatable_registers())
