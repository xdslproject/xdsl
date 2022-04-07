from io import StringIO

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


def test_rise_rewriting_fuse_dot():
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

    # after
    def get_rise_dsl_dot_fused(ctx: MLContext, builtin: Builtin, std: Std,
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
                    initializer := rise.literal(0, i32),
                    reductionLambda := rise._lambda(Block.from_callable([rise.tuple(rise.scalar(f32), rise.scalar(f32)), rise.scalar(f32)], lambda tuple, acc: [
                        fstFun := rise.fst(rise.scalar(f32), rise.scalar(f32)),
                        fst := rise.apply(fstFun, tuple),
                        sndFun := rise.snd(rise.scalar(f32), rise.scalar(f32)),
                        snd := rise.apply(sndFun, tuple),

                        result := rise.embed(fst, snd, acc, resultType=rise.scalar(f32), block=Block.from_callable([f32, f32, f32], lambda f, s, acc: [
                            product := arith.mulf(f, s),
                            result := arith.addf(product, acc),
                            rise._return(result),
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

    @dataclass
    class FuseReduceMap(RewritePattern):
        rise: Rise
        rise_dsl: RiseBuilder

        # def match_and_rewrite(self, op: Operation) -> RewriteResult[List[Operation] ~ Program]:
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
                #TODO: return this

                result = rise_dsl.reduce(
                    initializer, mapInput, [mapFun.s, reduceFun.t],
                    lambda tuple, acc: [
                        mapped := rise_dsl.apply(mapLambda, tuple),
                        result := rise_dsl.apply(reduceLambda, mapped, acc),
                        rise_dsl._return(result),
                    ])

                rewriter.replace_matched_op(result)

                #TODO: garbage collect
                rewriter.erase_op(reduceFun)
                rewriter.erase_op(applyMap)
                rewriter.erase_op(mapFun)

    @dataclass
    class BetaReduction(RewritePattern):
        rise: Rise
        rise_dsl: RiseBuilder

        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(lambdaApply := op, Apply) and isinstance(
                    _lambda := lambdaApply.fun.op, Lambda):
                print("performing betareduction")
                # substitution
                for i, block_arg in enumerate(_lambda.body.blocks[0].args):
                    block_arg.replace_by(lambdaApply.args[i])

                lambdaReturn = _lambda.body.blocks[0].ops[-1]
                op.results[0].replace_by(lambdaReturn.value.op.results[0])

                #inline
                rewriter.inline_block_before_matched_op(_lambda.body.blocks[0])

                #cleanup
                rewriter.erase_op(op)
                rewriter.erase_op(_lambda)
                rewriter.erase_op(lambdaReturn)

    @dataclass
    class FuseEmbeds(RewritePattern):
        rise: Rise
        rise_dsl: RiseBuilder

        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(secondEmbed := op, Embed) and any(
                    isinstance(arg, OpResult)
                    and isinstance(firstEmbed := arg.op, Embed)
                    and len(firstEmbed.results[0].uses) == 1
                    for arg in op.operands):
                print("performing fusion of two Embed operations")

                # replace use of result of first Embed with the actual result from its region
                embedReturn = firstEmbed.body.blocks[0].ops[-1]
                index = secondEmbed.operands.index(firstEmbed.results[0])
                secondEmbed.body.blocks[0].args[index].replace_by(
                    embedReturn.value.op.results[0])

                # build new embed op with according number of operands and blockargs
                operands = list(firstEmbed.operands)
                operands.extend(secondEmbed.operands)
                operands.remove(firstEmbed.results[0])

                blockArgs = list(arg.typ
                                 for arg in firstEmbed.body.blocks[0].args)
                secondBlockArgs = list(
                    arg.typ for arg in secondEmbed.body.blocks[0].args)
                del secondBlockArgs[index]
                blockArgs.extend(secondBlockArgs)

                newEmbed = Embed.create(
                    operands, [secondEmbed.output.typ], [], [], [
                        Region.from_block_list(
                            [Block.from_arg_types(blockArgs)])
                    ])

                # replace uses of block arguments while dropping use of the firstEmbed
                for i, arg in enumerate(firstEmbed.body.blocks[0].args):
                    arg.replace_by(newEmbed.body.blocks[0].args[i])

                firstEmbedArgNum = len(firstEmbed.body.blocks[0].args)
                offset = 0
                for i, arg in enumerate(secondEmbed.body.blocks[0].args):
                    if i == index:
                        offset = -1
                        continue
                    arg.replace_by(
                        newEmbed.body.blocks[0].args[firstEmbedArgNum + i +
                                                     offset])

                # move embedded operations of both old embeds to the new embed operation
                rewriter.inline_block_at_pos(secondEmbed.body.blocks[0],
                                             newEmbed.body.blocks[0], 0)

                rewriter.inline_block_at_pos(firstEmbed.body.blocks[0],
                                             newEmbed.body.blocks[0], 0)

                # cleanup
                rewriter.replace_matched_op(newEmbed)
                rewriter.erase_op(firstEmbed)
                rewriter.erase_op(embedReturn)

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
    PatternRewriteWalker(BetaReduction(rise, rise_dsl)).rewrite_module(unfused)
    PatternRewriteWalker(FuseEmbeds(rise, rise_dsl),
                         apply_recursively=False).rewrite_module(unfused)

    printer = Printer()
    printer.print_op(unfused)

    fused = get_rise_dsl_dot_fused(ctx, builtin, std, arith, affine, rise)
    fused.verify()

    file_unfused = StringIO("")
    printer = Printer(stream=file_unfused)
    printer.print_op(unfused)

    printer = Printer()
    printer.print_op(fused)

    file_fused = StringIO("")
    printer_dsl = Printer(stream=file_fused)
    printer_dsl.print_op(fused)

    assert file_unfused.getvalue().strip() == file_fused.getvalue().strip()


if __name__ == "__main__":
    test_rise_rewriting_fuse_dot()