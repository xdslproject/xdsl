import pytest

from xdsl.dialects import x86
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue


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


def test_push_op():
    r1 = TestSSAValue(x86.register.RDX)
    push_op_wrong = x86.PushOp(destination=x86.register.RDX)
    with pytest.raises(VerifyException):
        push_op_wrong.verify()
    push_op_wrong2 = x86.PushOp(source=r1, destination=x86.register.RAX)
    with pytest.raises(VerifyException):
        push_op_wrong2.verify()
    push_op = x86.PushOp(source=r1)
    push_op.verify()

    assert push_op.assembly_line() == "    push rdx"


def test_pop_op():
    r1 = TestSSAValue(x86.register.RAX)
    pop_op_wrong = x86.PopOp(source=r1)
    with pytest.raises(VerifyException):
        pop_op_wrong.verify()
    pop_op_wrong2 = x86.PopOp(r1, destination=x86.register.RDX)
    with pytest.raises(VerifyException):
        pop_op_wrong2.verify()
    pop_op = x86.PopOp(destination=x86.register.RDX)
    pop_op.verify()

    assert pop_op.assembly_line() == "    pop rdx"


def test_idiv_op():
    r1 = TestSSAValue(x86.register.RAX)
    idiv_op_wrong1 = x86.IdivOp(source=r1, destination=x86.register.RAX)
    with pytest.raises(VerifyException):
        idiv_op_wrong1.verify()
    idiv_op_wrong2 = x86.IdivOp(destination=x86.register.RAX)
    with pytest.raises(VerifyException):
        idiv_op_wrong2.verify()
    idiv_op = x86.IdivOp(source=r1)
    idiv_op.verify()

    assert idiv_op.assembly_line() == "    idiv rax"


def test_not_op():
    r1 = TestSSAValue(x86.register.RAX)
    not_op_wrong1 = x86.NotOp(source=r1)
    with pytest.raises(VerifyException):
        not_op_wrong1.verify()
    not_op_wrong2 = x86.NotOp(destination=x86.register.RAX)
    with pytest.raises(VerifyException):
        not_op_wrong2.verify()
    not_op = x86.NotOp(source=r1, destination=x86.register.RAX)
    not_op.verify()

    assert not_op.assembly_line() == "    not rax"
