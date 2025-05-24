from xdsl.backend.x86.register_queue import X86RegisterQueue
from xdsl.dialects import x86


def test_default_reserved_registers():
    register_queue = X86RegisterQueue.default()

    for reg in (
        x86.register.RAX,
        x86.register.RDX,
        x86.register.RSP,
    ):
        available_before = register_queue.available_registers.copy()
        register_queue.push(reg)
        assert available_before == register_queue.available_registers


def test_push_infinite_register():
    register_queue = X86RegisterQueue()

    infinite0 = x86.GeneralRegisterType.infinite_register(0)
    register_queue.push(infinite0)
    assert register_queue.pop(x86.GeneralRegisterType) == infinite0


def test_push_register():
    register_queue = X86RegisterQueue()

    register_queue.push(x86.register.RBX)
    assert register_queue.pop(x86.GeneralRegisterType) == x86.register.RBX

    register_queue.push(x86.register.YMM0)
    assert register_queue.pop(x86.AVX2RegisterType) == x86.register.YMM0
