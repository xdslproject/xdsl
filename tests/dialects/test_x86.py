from xdsl.dialects import x86
from xdsl.utils.test_value import TestSSAValue


def test_add_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    add_op = x86.AddOp(r1, r2, result=x86.Registers.RAX)

    print(add_op.assembly_line())


def test_mov_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    mov_op = x86.MovOp(r1, r2, result=x86.Registers.RAX)

    print(mov_op.assembly_line())


def test_sub_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    sub_op = x86.SubOp(r1, r2, result=x86.Registers.RAX)

    print(sub_op.assembly_line())


def test_imul_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    imul_op = x86.ImulOp(r1, r2, result=x86.Registers.RAX)

    print(imul_op.assembly_line())


def test_idiv_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    idiv_op = x86.IdivOp(
        r1,
    )

    print(idiv_op.assembly_line())


def test_and_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    and_op = x86.AndOp(r1, r2, result=x86.Registers.RAX)

    print(and_op.assembly_line())


def test_or_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    or_op = x86.OrOp(r1, r2, result=x86.Registers.RAX)

    print(or_op.assembly_line())


def test_xor_op():
    r1 = TestSSAValue(x86.Registers.RAX)
    r2 = TestSSAValue(x86.Registers.RDX)
    xor_op = x86.XorOp(r1, r2, result=x86.Registers.RAX)

    print(xor_op.assembly_line())


def test_not_op():
    r1 = TestSSAValue(x86.Registers.RDX)
    not_op = x86.NotOp(r1, destination=x86.Registers.RDX)

    print(not_op.assembly_line())


def test_push_op():
    r1 = TestSSAValue(x86.Registers.RDX)
    push_op = x86.PushOp(r1)

    print(push_op.assembly_line())


def test_pop_op():
    pop_op = x86.PopOp(destination=x86.Registers.RDX)

    print(pop_op.assembly_line())
