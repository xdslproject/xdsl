import pytest

from xdsl.backend.register_stack import OutOfRegisters
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
