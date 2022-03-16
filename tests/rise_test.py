from xdsl.dialects.affine import Affine
from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.std import *
from xdsl.dialects.std import Return as stdReturn
from xdsl.dialects.arith import *
from xdsl.dialects.rise import *


def get_rise_dot(ctx: MLContext, builtin: Builtin, std: Std, arith: Arith,
                 affine: Affine, rise: Rise) -> Operation:

    def fun(arg0: BlockArgument, arg1: BlockArgument,
            arg2: BlockArgument) -> List[Operation]:
        # yapf: disable
        return [
            rise.lowering_unit(Block.from_ops([
                in0 := rise.inOp(arg0, rise.array(rise.nat(5), rise.scalar(f32))),
                in1 := rise.inOp(arg1, rise.array(rise.nat(5), rise.scalar(f32))),
                zipFun := rise.zip(rise.nat(5), rise.scalar(f32), rise.scalar(f32)),
                zipped := rise.apply(zipFun, in0, in1),
                reductionLambda := rise._lambda(rise.fun(rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.fun(rise.scalar(f32), rise.scalar(f32))), Block.from_callable([rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.scalar(f32)], lambda tuple, acc: [
                    fstFun := rise.fst(rise.scalar(f32), rise.scalar(f32)),
                    sndFun := rise.snd(rise.scalar(f32), rise.scalar(f32)),

                    fst := rise.apply(fstFun, tuple),
                    snd := rise.apply(sndFun, tuple),

                    result := rise.embed(fst, snd, acc, resultType=rise.scalar(f32), block=Block.from_callable([f32, f32, f32], lambda f, s, acc: [
                        product := arith.mulf(f, s),
                        result := arith.addf(product, acc),
                        rise._return(result)
                    ]))
                ])),
                initializer := rise.literal(0, i32),
                reduceFun := rise.reduce(rise.nat(5), rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.scalar(f32)),
                result := rise.apply(reduceFun, reductionLambda, initializer, zipped),
                rise.out(result, arg2)
            ])),
            stdReturn.get()
        ]
    # yapf: enable

    f = FuncOp.from_callable("fun", [
        ArrayAttr.from_list([f32, f32, f32, f32, f32]),
        ArrayAttr.from_list([f32, f32, f32, f32, f32]), f32
    ], [], fun)
    return f


def test_rise_dot():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = Rise(ctx)
    affine = Affine(ctx)

    f = get_rise_dot(ctx, builtin, std, arith, affine, rise)
    f.verify()
    printer = Printer()
    printer.print_op(f)


if __name__ == "__main__":
    test_rise_dot()
