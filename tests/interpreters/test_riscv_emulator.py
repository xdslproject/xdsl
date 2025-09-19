from contextlib import redirect_stdout
from io import StringIO

import pytest

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import riscv, riscv_debug, riscv_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import BlockArgument
from xdsl.transforms.riscv_allocate_registers import RISCVAllocateRegistersPass

pytest.importorskip("riscemu", reason="riscemu is an optional dependency")

from xdsl.interpreters.riscv_emulator import RV_Debug, run_riscv  # noqa: E402

ctx = Context()
ctx.load_dialect(riscv.RISCV)


def test_simple():
    @ModuleOp
    @Builder.implicit_region
    def module():
        @Builder.implicit_region
        def body():
            six = riscv.LiOp(6).rd
            seven = riscv.LiOp(7).rd
            forty_two = riscv.MulOp(six, seven).rd
            riscv_debug.PrintfOp("{}", (forty_two,))
            riscv.ReturnOp()

        riscv_func.FuncOp("main", body, ((), ()), visibility="public")

    RISCVAllocateRegistersPass().apply(ctx, module)

    code = riscv.riscv_code(module)

    with StringIO() as stream, redirect_stdout(stream):
        run_riscv(
            code,
            extensions=[RV_Debug],
            unlimited_regs=True,
            verbosity=0,
        )
        assert stream.getvalue() == "42\n"


def test_multiply_add():
    @ModuleOp
    @Builder.implicit_region
    def module():
        @Builder.implicit_region
        def main():
            riscv.LiOp(3, rd=riscv.Registers.A0)
            riscv.LiOp(2, rd=riscv.Registers.A1)
            riscv.LiOp(1, rd=riscv.Registers.A2)

            riscv.JalOp("muladd")
            res = riscv.GetRegisterOp(riscv.Registers.A0).res
            riscv_debug.PrintfOp("{}", (res,))

            riscv.LiOp(93, rd=riscv.Registers.A7)
            riscv.EcallOp()

        riscv_func.FuncOp("main", main, ((), ()), visibility="public")

        @Builder.implicit_region((riscv.Registers.A0, riscv.Registers.A1))
        def multiply(args: tuple[BlockArgument, ...]):
            riscv.CommentOp("no extra registers needed, so no need to deal with stack")
            rs1, rs2 = args
            res = riscv.MulOp(rs1, rs2, rd=riscv.Registers.A0).rd
            riscv_func.ReturnOp(res)

        riscv_func.FuncOp(
            "multiply",
            multiply,
            ((riscv.Registers.A0, riscv.Registers.A1), (riscv.Registers.A0,)),
        )

        @Builder.implicit_region((riscv.Registers.A0, riscv.Registers.A1))
        def add(args: tuple[BlockArgument, ...]):
            riscv.CommentOp("no extra registers needed, so no need to deal with stack")
            rs1, rs2 = args
            res = riscv.AddOp(rs1, rs2, rd=riscv.Registers.A0).rd
            riscv_func.ReturnOp(res)

        riscv_func.FuncOp(
            "add",
            add,
            ((riscv.Registers.A0, riscv.Registers.A1), (riscv.Registers.A0,)),
        )

        @Builder.implicit_region(
            (riscv.Registers.A0, riscv.Registers.A1, riscv.Registers.A2)
        )
        def muladd(args: tuple[BlockArgument, ...]):
            riscv.CommentOp("a0 <- a0 * a1 + a2")
            riscv.CommentOp("prologue")
            # get registers with the arguments to muladd
            _, _, a2_muladd = args

            # get registers we'll use in this section
            sp_muladd = riscv.GetRegisterOp(riscv.Registers.SP)
            s0_muladd_0 = riscv.GetRegisterOp(riscv.Registers.S0)
            ra_muladd = riscv.GetRegisterOp(riscv.Registers.RA)
            riscv.CommentOp(
                "decrement stack pointer by number of register values we need to store for later"
            )
            riscv.AddiOp(sp_muladd, -8, rd=riscv.Registers.SP)
            riscv.CommentOp("save the s registers we'll use on the stack")
            riscv.SwOp(sp_muladd, s0_muladd_0, 0)
            riscv.CommentOp("save the return address we'll use on the stack")
            riscv.SwOp(sp_muladd, ra_muladd, 4)

            # store third parameter, in a2 to the temporary register s0
            # guaranteed to be the same after call to multiply
            s0_muladd_1 = riscv.MVOp(a2_muladd, rd=riscv.Registers.S0)
            riscv.JalOp("multiply")

            # The product of a0 and a1 is stored in a0
            # We now have to move s0 to a1, and call add

            riscv.MVOp(s0_muladd_1, rd=riscv.Registers.A1)

            riscv.JalOp("add")

            riscv.CommentOp("epilogue")
            riscv.CommentOp("store the old values back into the s registers")
            riscv.LwOp(sp_muladd, 0, rd=riscv.Registers.S0)

            riscv.CommentOp("store the return address back into the ra register")
            riscv.LwOp(sp_muladd, 4, rd=riscv.Registers.RA)

            riscv.CommentOp(
                "set the sp back to what it was at the start of the function call"
            )
            riscv.AddiOp(sp_muladd, 8, rd=riscv.Registers.SP)

            riscv.CommentOp("jump back to caller")
            riscv.ReturnOp()

        riscv_func.FuncOp(
            "muladd",
            muladd,
            (
                (riscv.Registers.A0, riscv.Registers.A1, riscv.Registers.A2),
                (riscv.Registers.A0,),
            ),
        )

    RISCVAllocateRegistersPass().apply(ctx, module)

    code = riscv.riscv_code(module)
    with StringIO() as stream, redirect_stdout(stream):
        run_riscv(
            code,
            extensions=[RV_Debug],
            unlimited_regs=True,
            verbosity=0,
        )
        assert stream.getvalue() == "7\n"
