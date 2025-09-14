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
    register_stack = X86RegisterStack()

    infinite0 = x86.AVX2RegisterType.infinite_register(0)
    register_stack.push(infinite0)
    assert register_stack.pop(x86.AVX2RegisterType) == infinite0


def test_push_register():
    register_stack = X86RegisterStack.get()

    register_stack.push(x86.registers.YMM0)
    assert register_stack.pop(x86.AVX2RegisterType) == x86.registers.YMM0

    register_stack.push(x86.registers.RAX)
    assert register_stack.pop(x86.GeneralRegisterType) == x86.registers.RAX
