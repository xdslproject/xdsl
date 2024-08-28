from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.csl import csl, csl_wrapper
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class RemoveUnusedYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.YieldOp, rewriter: PatternRewriter, /):
        parent = op.parent_op()
        assert isa(parent, csl_wrapper.ModuleOp)
        if op.parent_region() == parent.program_module:
            rewriter.erase_matched_op()


@dataclass(frozen=True)
class ExtractLayoutProgramModule(RewritePattern):
    prog_name: str

    @staticmethod
    def replace_arg(
        rewriter: PatternRewriter, arg: BlockArgument, param: Operation | SSAValue
    ):
        for use in arg.uses:
            rewriter.handle_operation_modification(use.operation)
        arg.replace_by(SSAValue.get(param))
        rewriter.erase_block_argument(arg)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /):
        module_block = Block()

        outer_loop_block = Block()
        outer_loop_block.insert_arg(builtin.IntegerType(32), 0)
        x = outer_loop_block.args[0]

        inner_loop_block = Block()
        inner_loop_block.insert_arg(builtin.IntegerType(32), 0)
        y = inner_loop_block.args[0]

        new_args = list[SSAValue]()

        assert isa(yield_op := op.layout_module.block.last_op, csl_wrapper.YieldOp)
        yield_op.detach()

        with ImplicitBuilder(module_block):
            const_0 = arith.Constant.from_int_and_width(0, builtin.IndexType())
            const_1 = arith.Constant.from_int_and_width(1, builtin.IndexType())

            const_width = arith.Constant(op.width)
            param_width = csl.ParamOp("width", op.width.type, const_width)

            const_height = arith.Constant(op.height)
            param_height = csl.ParamOp("height", op.height.type, const_height)

            for param in op.params:
                if isa(param.value, builtin.IntegerAttr):
                    value = arith.Constant(param.value)
                else:
                    value = None
                p = csl.ParamOp(param.key.data, param.type, value)
                new_args.append(SSAValue.get(p))

            layout = csl.LayoutOp(Region())
            with ImplicitBuilder(layout.body.block):
                scf.For(
                    lb=const_0,
                    ub=const_width,
                    step=const_1,
                    iter_args=[x],
                    body=outer_loop_block,
                )

                with ImplicitBuilder(outer_loop_block):
                    scf.For(
                        lb=const_0,
                        ub=const_height,
                        step=const_1,
                        iter_args=[y],
                        body=inner_loop_block,
                    )
        rewriter.inline_block(
            # TODO(dk949): the clone is temporary
            op.layout_module.block,
            # op.layout_module.clone().block,
            InsertPoint.at_start(inner_loop_block),
            arg_values=[
                SSAValue.get(param_width),
                SSAValue.get(param_height),
                SSAValue.get(x),
                SSAValue.get(y),
                *new_args,
            ],
        )
        tile_code = csl.SetTileCodeOp(
            fname=self.prog_name,
            x_coord=x,
            y_coord=y,
        )
        inner_loop_block.add_op(tile_code)

        layout_mod = csl.CslModuleOp(
            regions=[Region(module_block)],
            properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.LAYOUT)},
        )
        print(layout_mod)
        exit(0)


@dataclass(frozen=True)
class CslWrapperToCslPass(ModulePass):
    """
    Wraps program in the csl_stencil dialect in a csl_wrapper by translating each
    top-level function to one module wrapper.
    """

    name = "csl-wrapper-to-csl"
    prog_name: str

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveUnusedYield(),
                    ExtractLayoutProgramModule(prog_name=self.prog_name),
                ]
            ),
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
