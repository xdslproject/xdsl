from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import func, riscv, riscv_snitch, snitch
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.riscv_snitch import RiscvSnitchFunctions
from xdsl.interpreters.utils.stream import Acc, Nats
from xdsl.ir import BlockArgument
from xdsl.utils.test_value import create_ssa_value


def test_read_write():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvSnitchFunctions())

    a0 = riscv.Registers.A0
    a1 = riscv.Registers.A1

    input_stream = Nats()
    output_stream = Acc()

    assert interpreter.run_op(
        riscv_snitch.ReadOp(create_ssa_value(snitch.ReadableStreamType(a0)), a0),
        (input_stream,),
    ) == (1,)
    assert interpreter.run_op(
        riscv_snitch.ReadOp(create_ssa_value(snitch.ReadableStreamType(a1)), a1),
        (input_stream,),
    ) == (2,)

    assert (
        interpreter.run_op(
            riscv_snitch.WriteOp(
                create_ssa_value(a0), create_ssa_value(snitch.ReadableStreamType(a0))
            ),
            (
                1,
                output_stream,
            ),
        )
        == ()
    )
    assert output_stream.values == [1]
    assert (
        interpreter.run_op(
            riscv_snitch.WriteOp(
                create_ssa_value(a1), create_ssa_value(snitch.ReadableStreamType(a1))
            ),
            (
                2,
                output_stream,
            ),
        )
        == ()
    )
    assert output_stream.values == [1, 2]


def test_frep_carried_vars():
    float_register = riscv.Registers.UNALLOCATED_FLOAT
    acc_reg_type = riscv.Registers.FT[3]

    @ModuleOp
    @Builder.implicit_region
    def sum_to_for_op():
        with ImplicitBuilder(
            func.FuncOp("sum_to", ((float_register,), (float_register,))).body
        ) as (count,):
            one = riscv.LiOp(1).rd
            initial = riscv.FCvtDWOp(one, rd=acc_reg_type).rd

            @Builder.implicit_region((float_register,))
            def for_loop_region(args: tuple[BlockArgument, ...]):
                (acc,) = args
                res = riscv.FAddDOp(acc, acc, rd=acc_reg_type)
                riscv_snitch.FrepYieldOp(res)

            result = riscv_snitch.FrepOuterOp(count, for_loop_region, (initial,)).res
            func.ReturnOp(*result)

    interpreter = Interpreter(sum_to_for_op)
    interpreter.register_implementations(RiscvSnitchFunctions())
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(FuncFunctions())

    assert interpreter.call_op("sum_to", (5,)) == (64,)
