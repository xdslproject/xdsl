from xdsl.dialects import x86
from xdsl.utils.test_value import TestSSAValue


def test_add_op():
    rs = TestSSAValue(x86.Registers.RDX)
    add_op = x86.AddOp(rs, rd=x86.Registers.RAX)

    print(add_op.assembly_line())


def test_sub_op():
    rs = TestSSAValue(x86.Registers.RDX)
    sub_op = x86.SubOp(rs, rd=x86.Registers.RAX)

    print(sub_op.assembly_line())


def test_imul_op():
    rs = TestSSAValue(x86.Registers.RDX)
    imul_op = x86.ImulOp(rs, rd=x86.Registers.RAX)

    print(imul_op.assembly_line())


def test_idiv_op():
    rs = TestSSAValue(x86.Registers.RDX)
    idiv_op = x86.IdivOp(rs)

    print(idiv_op.assembly_line())


def test_mov_op():
    rs = TestSSAValue(x86.Registers.RDX)
    mov_op = x86.MovOp(rs, rd=x86.Registers.RAX)

    print(mov_op.assembly_line())


def test_push_op():
    rs = TestSSAValue(x86.Registers.RDX)
    push_op = x86.PushOp(rs)

    print(push_op.assembly_line())


def test_pop_op():
    pop_op = x86.PopOp(rd=x86.Registers.RDX)

    print(pop_op.assembly_line())


def test_and_op():
    rs = TestSSAValue(x86.Registers.RDX)
    and_op = x86.AndOp(rs, rd=x86.Registers.RAX)

    print(and_op.assembly_line())


def test_or_op():
    rs = TestSSAValue(x86.Registers.RDX)
    or_op = x86.OrOp(rs, rd=x86.Registers.RAX)

    print(or_op.assembly_line())


def test_xor_op():
    rs = TestSSAValue(x86.Registers.RDX)
    xor_op = x86.XorOp(rs, rd=x86.Registers.RAX)

    print(xor_op.assembly_line())


def test_not_op():
    not_op = x86.NotOp(rd=x86.Registers.RAX)

    print(not_op.assembly_line())
