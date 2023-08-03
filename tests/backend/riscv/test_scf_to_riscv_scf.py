from xdsl.backend.riscv.lowering.scf_to_riscv_scf import ScfToRiscvPass
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, riscv, riscv_scf, scf
from xdsl.ir import BlockArgument, MLContext

INDEX_TYPE = builtin.IndexType()
REGISTER_TYPE = riscv.IntRegisterType.unallocated()


def test_lower_simple_scf_for():
    @builtin.ModuleOp
    @Builder.implicit_region
    def simple_scf_for():
        lb = arith.Constant.from_int_and_width(0, INDEX_TYPE)
        ub = arith.Constant.from_int_and_width(10, INDEX_TYPE)
        step = arith.Constant.from_int_and_width(1, INDEX_TYPE)
        initial = arith.Constant.from_int_and_width(0, INDEX_TYPE)

        @Builder.implicit_region((INDEX_TYPE, INDEX_TYPE))
        def for_loop_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            res = arith.Addi(i, acc)
            scf.Yield.get(res)

        scf.For.get(lb, ub, step, (initial,), for_loop_region)

    @builtin.ModuleOp
    @Builder.implicit_region
    def expected():
        lb = arith.Constant.from_int_and_width(0, INDEX_TYPE)
        ub = arith.Constant.from_int_and_width(10, INDEX_TYPE)
        step = arith.Constant.from_int_and_width(1, INDEX_TYPE)
        initial = arith.Constant.from_int_and_width(0, INDEX_TYPE)
        lb_cast = builtin.UnrealizedConversionCastOp.get((lb,), (REGISTER_TYPE,))
        ub_cast = builtin.UnrealizedConversionCastOp.get((ub,), (REGISTER_TYPE,))
        step_cast = builtin.UnrealizedConversionCastOp.get((step,), (REGISTER_TYPE,))
        initial_cast = builtin.UnrealizedConversionCastOp.get(
            (initial,), (REGISTER_TYPE,)
        )

        @Builder.implicit_region((REGISTER_TYPE, REGISTER_TYPE))
        def for_loop_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            acc_cast = builtin.UnrealizedConversionCastOp.get((acc,), (INDEX_TYPE,))
            i_cast = builtin.UnrealizedConversionCastOp.get((i,), (INDEX_TYPE,))
            res = arith.Addi(i_cast, acc_cast)
            res_cast = builtin.UnrealizedConversionCastOp((res,), (REGISTER_TYPE,))
            riscv_scf.YieldOp(res_cast)

        res = riscv_scf.ForOp(
            lb_cast, ub_cast, step_cast, (initial_cast,), for_loop_region
        )
        builtin.UnrealizedConversionCastOp.get((res,), (INDEX_TYPE,))

    # check that the lowered region is still valid
    expected.verify()

    ScfToRiscvPass().apply(MLContext(), simple_scf_for)
    assert f"{expected}" == f"{simple_scf_for}"
