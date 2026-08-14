import re

import pytest

from xdsl.backend.register_stack import OutOfRegisters
from xdsl.backend.x86 import arch
from xdsl.backend.x86.register_stack import X86RegisterStack
from xdsl.dialects import x86
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i32
from xdsl.dialects.x86 import registers
from xdsl.utils.exceptions import DiagnosticException


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


def test_allocatable_registers_follow_the_target():
    """
    xmm16-31 and ymm16-31 need EVEX, so they are only allocatable on AVX-512.
    Handing them out on a VEX-only target produces assembly that faults with #UD
    on hardware that does not have them.
    """

    def vector_indices(stack: X86RegisterStack) -> set[int]:
        return stack.allocatable_registers[registers.X86_VECTOR_POOL_KEY]

    assert vector_indices(X86RegisterStack.get_for_arch(arch.AVX2)) == set(range(16))
    assert vector_indices(X86RegisterStack.get_for_arch(arch.UNKNOWN)) == set(range(16))
    assert vector_indices(X86RegisterStack.get_for_arch(arch.AVX512)) == set(range(32))

    # The default stack, used when no target is recorded, is the conservative one.
    assert vector_indices(X86RegisterStack.get()) == set(range(16))


def test_arch_round_trips_through_the_module():
    module = ModuleOp([])
    assert arch.X86Arch.from_module(module) is arch.UNKNOWN

    arch.AVX512.set_on_module(module)
    assert module.attributes[arch.ARCH_ATTR_NAME] == StringAttr("avx512")
    assert arch.X86Arch.from_module(module).name() == "avx512"

    arch.AVX2.set_on_module(module)
    assert arch.X86Arch.from_module(module).name() == "avx2"


def test_arch_on_module_must_be_a_string():
    module = ModuleOp([])
    module.attributes[arch.ARCH_ATTR_NAME] = IntegerAttr(2, i32)
    with pytest.raises(DiagnosticException, match=re.escape("must be a string")):
        arch.X86Arch.from_module(module)
