from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.std import *
from xdsl.dialects.arith import *
from xdsl.dialects.rise import *

#%zipFun : !rise.fun<!scalar<!i32>, !scalar<!i32>> = rise.zip() ["n" = !nat<1>, "s" = !scalar<!i32>, "t" = !scalar<!i32>]


def get_rise_vector_mul(ctx: MLContext, builtin: Builtin, std: Std,
                        rise: Rise) -> Operation:

    def fun(arg0: BlockArgument, arg1: BlockArgument) -> List[Operation]:
        # yapf: disable
        return [
            in0 := rise.inOp(arg0, rise.array(rise.nat(5), rise.scalar(f32))),
            Return.get(arg1)
        ]
    # yapf: enable

    f = FuncOp.from_callable("fun", [
        ArrayAttr.from_list([f32, f32, f32, f32, f32]),
        ArrayAttr.from_list([f32, f32, f32, f32, f32])
    ], [f32], fun)
    return f


def test_affine():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = Rise(ctx)

    f = get_rise_vector_mul(ctx, builtin, std, rise)
    f.verify()
    printer = Printer()
    printer.print_op(f)


if __name__ == "__main__":
    test_affine()
