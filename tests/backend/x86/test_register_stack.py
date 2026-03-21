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
    assert register_stack.pop(x86.GeneralRegisterType) == x86.registers.RAX


def test_vector_registers_share_one_pool():
    """
    All x86 vector kinds share one pool key; indices 0..15 alias across xmm/ymm/zmm.
    After every ABI index usable by SSE is taken, AVX2 cannot allocate (same indices).
    """
    stack = X86RegisterStack.get()
    pool_key = x86.registers.X86_VECTOR_POOL_KEY
    assert len(stack.available_registers[pool_key]) == 32
    n_sse = len(x86.SSERegisterType.abi_name_by_index())
    for _ in range(n_sse):
        stack.pop(x86.SSERegisterType)
    with pytest.raises(OutOfRegisters):
        stack.pop(x86.AVX2RegisterType)


def test_vector_pool_includes_zmm16_through_zmm31():
    """Default stack seeds zmm0-zmm31; indices 16-31 share the same pool key as low vector."""
    stack = X86RegisterStack.get()
    pool_key = x86.registers.X86_VECTOR_POOL_KEY
    assert len(stack.available_registers[pool_key]) == 32
    for _ in range(32):
        stack.pop(x86.AVX512RegisterType)
    with pytest.raises(OutOfRegisters):
        stack.pop(x86.AVX512RegisterType)
