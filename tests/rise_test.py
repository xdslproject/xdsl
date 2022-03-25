from io import StringIO
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
                initializer := rise.literal(0, i32),
                zipFun := rise.zip(rise.nat(5), rise.scalar(f32), rise.scalar(f32)),
                zipped := rise.apply(zipFun, in0, in1),
                reductionLambda := rise._lambda(Block.from_callable([rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.scalar(f32)], lambda tuple, acc: [
                    fstFun := rise.fst(rise.scalar(f32), rise.scalar(f32)),
                    fst := rise.apply(fstFun, tuple),
                    sndFun := rise.snd(rise.scalar(f32), rise.scalar(f32)),
                    snd := rise.apply(sndFun, tuple),

                    result := rise.embed(fst, snd, acc, resultType=rise.scalar(f32), block=Block.from_callable([f32, f32, f32], lambda f, s, acc: [
                        product := arith.mulf(f, s),
                        result := arith.addf(product, acc),
                        stdReturn.get(result),
                    ])),
                    rise._return(result),
                ])),
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


def get_rise_dsl_dot(ctx: MLContext, builtin: Builtin, std: Std, arith: Arith,
                     affine: Affine, rise: RiseBuilder) -> Operation:

    def fun(arg0: BlockArgument, arg1: BlockArgument,
            arg2: BlockArgument) -> List[Operation]:
        # yapf: disable
        return [
            rise.lowering_unit(lambda: [
                in0 := rise.inOp(arg0, rise.array(rise.nat(5), rise.scalar(f32))),
                in1 := rise.inOp(arg1, rise.array(rise.nat(5), rise.scalar(f32))),
                rise.out(rise.reduce(rise.literal(0, i32), rise.zip(in0, in1), [rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.scalar(f32)], lambda tuple, acc: [
                    result := rise.embed(rise.fst(tuple), rise.snd(tuple), acc, resultType=rise.scalar(f32), block=Block.from_callable([f32, f32, f32], lambda f, s, acc: [
                        product := arith.mulf(f, s),
                        result := arith.addf(product, acc),
                        stdReturn.get(result),
                    ])),
                    rise._return(result),
                ]), arg2)
            ]),
            stdReturn.get()
        ]
    # yapf: enable

    f = FuncOp.from_callable("fun", [
        ArrayAttr.from_list([f32, f32, f32, f32, f32]),
        ArrayAttr.from_list([f32, f32, f32, f32, f32]), f32
    ], [], fun)
    return f


def test_rise_dsl_dot():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = RiseBuilder(ctx)
    affine = Affine(ctx)

    f = get_rise_dsl_dot(ctx, builtin, std, arith, affine, rise)
    f.verify()
    printer = Printer()
    printer.print_op(f)


def test_dot_equal():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = Rise(ctx)
    riseDSL = RiseBuilder(ctx)
    affine = Affine(ctx)

    dot = get_rise_dot(ctx, builtin, std, arith, affine, rise)
    dot_dsl = get_rise_dsl_dot(ctx, builtin, std, arith, affine, riseDSL)

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(dot)

    file_dsl = StringIO("")
    printer_dsl = Printer(stream=file_dsl)
    printer_dsl.print_op(dot_dsl)

    assert file.getvalue().strip() == file_dsl.getvalue().strip()


if __name__ == "__main__":
    test_rise_dot()
    test_rise_dsl_dot()
    test_dot_equal()