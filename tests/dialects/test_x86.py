import pytest

from xdsl.dialects import x86


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.register.RAX, "rax"),
        (x86.register.RCX, "rcx"),
        (x86.register.RDX, "rdx"),
        (x86.register.RBX, "rbx"),
        (x86.register.RSP, "rsp"),
        (x86.register.RBP, "rbp"),
        (x86.register.RSI, "rsi"),
        (x86.register.RDI, "rdi"),
        (x86.register.R8, "r8"),
        (x86.register.R9, "r9"),
        (x86.register.R10, "r10"),
        (x86.register.R11, "r11"),
        (x86.register.R12, "r12"),
        (x86.register.R13, "r13"),
        (x86.register.R14, "r14"),
        (x86.register.R15, "r15"),
    ],
)
def test_register(register: x86.register.GeneralRegisterType, name: str):
    assert register.is_allocated
    assert register.register_name == name


def test_rflags_register():
    rflags = x86.register.RFLAGS
    assert rflags.is_allocated
    assert rflags.register_name == "rflags"
