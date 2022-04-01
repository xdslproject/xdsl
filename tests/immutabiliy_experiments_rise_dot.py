from __future__ import annotations
from io import StringIO
from mimetypes import init
from pprint import pprint

from xdsl.dialects.affine import Affine
from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from xdsl.printer import Printer
from xdsl.dialects.std import *
from xdsl.dialects.std import Return as stdReturn
from xdsl.dialects.arith import *
from xdsl.dialects.rise.rise import *
from xdsl.dialects.rise.riseBuilder import RiseBuilder
from xdsl.elevate import *

import difflib


def rewriting_with_immutability_experiments():
    # before
    def get_rise_dsl_dot_unfused(ctx: MLContext, builtin: Builtin, std: Std,
                                 arith: Arith, affine: Affine,
                                 rise: Rise) -> Operation:

        def fun(arg0: BlockArgument, arg1: BlockArgument,
                arg2: BlockArgument) -> List[Operation]:
            # yapf: disable
            return [
                rise.lowering_unit(Block.from_ops([
                    in0 := rise.inOp(arg0, rise.array(rise.nat(5), rise.scalar(f32))),
                    in1 := rise.inOp(arg1, rise.array(rise.nat(5), rise.scalar(f32))),
                    zipFun := rise.zip(rise.nat(5), rise.scalar(f32), rise.scalar(f32)),
                    zipped := rise.apply(zipFun, in0, in1),
                    sumLambda := rise._lambda(Block.from_callable([rise.tuple(rise.scalar(f32), rise.scalar(f32))], lambda tuple: [
                        fstFun := rise.fst(rise.scalar(f32), rise.scalar(f32)),
                        fst := rise.apply(fstFun, tuple),
                        sndFun := rise.snd(rise.scalar(f32), rise.scalar(f32)),
                        snd := rise.apply(sndFun, tuple),
                        result := rise.embed(fst, snd, resultType=rise.scalar(f32), block=Block.from_callable([f32, f32], lambda f, s: [
                            result := arith.mulf(f, s),
                            rise._return(result),
                        ])),
                        rise._return(result),
                    ])),
                    mapFun := rise.map(rise.nat(5), rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.scalar(f32)),
                    summed := rise.apply(mapFun, sumLambda, zipped),

                    initializer := rise.literal(0, i32),
                    reductionLambda := rise._lambda(Block.from_callable([rise.scalar(f32), rise.scalar(f32)], lambda elem, acc: [
                        result := rise.embed(elem, acc, resultType=rise.scalar(f32), block=Block.from_callable([f32, f32], lambda elem, acc: [
                            result := arith.addf(elem, acc),
                            rise._return(result),
                        ])),
                        rise._return(result),
                    ])),
                    reduceFun := rise.reduce(rise.nat(5), rise.scalar(f32), rise.scalar(f32)),
                    result := rise.apply(reduceFun, reductionLambda, initializer, summed),
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

    expected_fused_dot = \
"""builtin.func() ["sym_name" = "fun", "type" = !fun<[[!f32, !f32, !f32, !f32, !f32], [!f32, !f32, !f32, !f32, !f32], !f32], []>, "sym_visibility" = "private"] {
^0(%0 : [!f32, !f32, !f32, !f32, !f32], %1 : [!f32, !f32, !f32, !f32, !f32], %2 : !f32):
  rise.lowering_unit() {
    %3 : !rise.array<!nat<5>, !scalar<!f32>> = rise.in(%0 : [!f32, !f32, !f32, !f32, !f32]) ["type" = !rise.array<!nat<5>, !scalar<!f32>>]
    %4 : !rise.array<!nat<5>, !scalar<!f32>> = rise.in(%1 : [!f32, !f32, !f32, !f32, !f32]) ["type" = !rise.array<!nat<5>, !scalar<!f32>>]
    %5 : !rise.fun<!rise.array<!nat<5>, !scalar<!f32>>, !rise.fun<!rise.array<!nat<5>, !scalar<!f32>>, !rise.array<!nat<5>, !tuple<!scalar<!f32>, !scalar<!f32>>>>> = rise.zip() ["n" = !nat<5>, "s" = !scalar<!f32>, "t" = !scalar<!f32>]
    %6 : !rise.array<!nat<5>, !tuple<!scalar<!f32>, !scalar<!f32>>> = rise.apply(%5 : !rise.fun<!rise.array<!nat<5>, !scalar<!f32>>, !rise.fun<!rise.array<!nat<5>, !scalar<!f32>>, !rise.array<!nat<5>, !tuple<!scalar<!f32>, !scalar<!f32>>>>>, %3 : !rise.array<!nat<5>, !scalar<!f32>>, %4 : !rise.array<!nat<5>, !scalar<!f32>>)
    %7 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !scalar<!f32>> = rise.lambda() {
    ^1(%8 : !tuple<!scalar<!f32>, !scalar<!f32>>):
      %9 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !scalar<!f32>> = rise.fst() ["s" = !scalar<!f32>, "t" = !scalar<!f32>]
      %10 : !scalar<!f32> = rise.apply(%9 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !scalar<!f32>>, %8 : !tuple<!scalar<!f32>, !scalar<!f32>>)
      %11 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !scalar<!f32>> = rise.snd() ["s" = !scalar<!f32>, "t" = !scalar<!f32>]
      %12 : !scalar<!f32> = rise.apply(%11 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !scalar<!f32>>, %8 : !tuple<!scalar<!f32>, !scalar<!f32>>)
      %13 : !scalar<!f32> = rise.embed(%10 : !scalar<!f32>, %12 : !scalar<!f32>) {
      ^2(%14 : !f32, %15 : !f32):
        %16 : !f32 = arith.mulf(%14 : !f32, %15 : !f32)
        rise.return(%16 : !f32)
      }
      rise.return(%13 : !scalar<!f32>)
    }
    %17 : !f32 = rise.literal() ["value" = 0 : !i32]
    %18 : !rise.fun<!scalar<!f32>, !rise.fun<!scalar<!f32>, !scalar<!f32>>> = rise.lambda() {
    ^3(%19 : !scalar<!f32>, %20 : !scalar<!f32>):
      %21 : !scalar<!f32> = rise.embed(%19 : !scalar<!f32>, %20 : !scalar<!f32>) {
      ^4(%22 : !f32, %23 : !f32):
        %24 : !f32 = arith.addf(%22 : !f32, %23 : !f32)
        rise.return(%24 : !f32)
      }
      rise.return(%21 : !scalar<!f32>)
    }
    %25 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !rise.fun<!scalar<!f32>, !scalar<!f32>>> = rise.lambda() {
    ^5(%26 : !tuple<!scalar<!f32>, !scalar<!f32>>, %27 : !scalar<!f32>):
      %28 : !scalar<!f32> = rise.apply(%7 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !scalar<!f32>>, %26 : !tuple<!scalar<!f32>, !scalar<!f32>>)
      %29 : !scalar<!f32> = rise.apply(%18 : !rise.fun<!scalar<!f32>, !rise.fun<!scalar<!f32>, !scalar<!f32>>>, %28 : !scalar<!f32>, %27 : !scalar<!f32>)
      rise.return(%29 : !scalar<!f32>)
    }
    %30 : !rise.fun<!rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !rise.fun<!scalar<!f32>, !scalar<!f32>>>, !rise.fun<!scalar<!f32>, !rise.fun<!rise.array<!nat<5>, !tuple<!scalar<!f32>, !scalar<!f32>>>, !scalar<!f32>>>> = rise.reduce() ["n" = !nat<5>, "s" = !tuple<!scalar<!f32>, !scalar<!f32>>, "t" = !scalar<!f32>]
    %31 : !scalar<!f32> = rise.apply(%30 : !rise.fun<!rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !rise.fun<!scalar<!f32>, !scalar<!f32>>>, !rise.fun<!scalar<!f32>, !rise.fun<!rise.array<!nat<5>, !tuple<!scalar<!f32>, !scalar<!f32>>>, !scalar<!f32>>>>, %25 : !rise.fun<!tuple<!scalar<!f32>, !scalar<!f32>>, !rise.fun<!scalar<!f32>, !scalar<!f32>>>, %17 : !f32, %6 : !rise.array<!nat<5>, !tuple<!scalar<!f32>, !scalar<!f32>>>)
    rise.out(%31 : !scalar<!f32>, %2 : !f32)
  }
  std.return()
}
"""

    # @dataclass
    # class ImmutableOperation(Operation):
    #     op: Operation

    @dataclass
    class ImmutableValueView:
        _value: SSAValue

    @dataclass
    class ImmutableOpResultView:
        _op: ImmutableOpView
        _result_index: int

    @dataclass
    class ImmutableOpView:
        _op: Operation
        name: str = field(default="", init=False)
        _operands: FrozenList[Union[SSAValue, ImmutableValueView]] = field(
            default_factory=FrozenList)
        results: List[ImmutableOpResultView] = field(default_factory=list)

        regions: List[Region] = field(default_factory=list)
        parent: Optional[Block] = field(default=None)

        @staticmethod
        def from_op(op: Operation) -> ImmutableOpView:
            return ImmutableOpResultView(op, op.name, op.operands, op.results,
                                         op.regions, op.parent)

    @dataclass
    class FuseReduceMap(RewritePattern):
        rise: Rise
        rise_dsl: RiseBuilder

        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(applyReduce := op, Apply) and isinstance(
                    reduceFun := applyReduce.fun.op,
                    Reduce) and (reduceLambda := applyReduce.args[0].op) and (
                        initializer := applyReduce.args[1].op) and isinstance(
                            applyMap :=
                            applyReduce.args[2].op, Apply) and isinstance(
                                mapFun := applyMap.fun.op,
                                Map) and (mapLambda :=
                                          applyMap.args[0].op) and (
                                              mapInput := applyMap.args[1].op):

                print("fuseReduceMap match")

                initializer, mapInput, mapFun, reduceFun, mapLambda, reduceLambda = match(
                    initializer, mapInput, mapFun, reduceFun, mapLambda,
                    reduceLambda)

                result = rise_dsl.reduce(
                    initializer, mapInput, [mapFun.s, reduceFun.t],
                    lambda tuple, acc: [
                        mapped := rise_dsl.apply(mapLambda, tuple),
                        result := rise_dsl.apply(reduceLambda, mapped, acc),
                        rise_dsl._return(result),
                    ])

                rewriter.replace_matched_op(result)

                # TODO: garbage collect
                rewriter.erase_op(reduceFun)
                rewriter.erase_op(applyMap)
                rewriter.erase_op(mapFun)

    def match(*args):
        return args

    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = Rise(ctx)
    rise_dsl = RiseBuilder(ctx)
    affine = Affine(ctx)

    unfused = get_rise_dsl_dot_unfused(ctx, builtin, std, arith, affine, rise)
    unfused.verify()

    PatternRewriteWalker(FuseReduceMap(rise, rise_dsl)).rewrite_module(unfused)

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(unfused)
    diff = list(difflib.Differ().compare(file.getvalue().splitlines(True),
                                         expected_fused_dot.splitlines(True)))
    print(''.join(diff))
    assert file.getvalue().strip() == expected_fused_dot.strip()

    # Now do rewriting

    # Do easiest thing first. Do a complete clone of everything


if __name__ == "__main__":
    rewriting_with_immutability_experiments()