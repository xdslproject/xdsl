from xdsl.dialects.builtin import *
from xdsl.dialects.std import *
from xdsl.printer import Printer
from xdsl.dialects.affine import *


def get_example_affine_program(ctx: MLContext, builtin: Builtin, std: Std,
                               affine: Affine) -> Operation:
    def affine_mm(arg0: BlockArgument, arg1: BlockArgument,
                  arg2: BlockArgument) -> List[Operation]:
        # yapf: disable
        return [
            affine.for_(0, 256, Block.from_callable([std.i64], lambda i: [
                affine.for_(0, 256, Block.from_callable([std.i64], lambda j: [
                    affine.for_(0, 250, Block.from_callable([std.i64], lambda k: [
                        l := affine.load(arg0, i, k),
                        r := affine.load(arg1, k, j),
                        o := affine.load(arg2, i, j),
                        m := Mulf.get(l, r),
                        a := Mulf.get(o, m),
                        affine.store(a, arg2, i, j)
                    ]))
                ]))
            ])),
            Return.get(arg2)
        ]
    # yapf: enable

    f32 = std.f32
    f = FuncOp.from_callable("affine_mm", [f32, f32, f32], [f32], affine_mm)
    return f


def test_affine():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    affine = Affine(ctx)

    test_empty = new_op("test_empty", 0, 0, 0)
    ctx.register_op(test_empty)
    op = test_empty()

    f = get_example_affine_program(ctx, builtin, std, affine)
    f.verify()
    printer = Printer()
    printer.print_op(f)
