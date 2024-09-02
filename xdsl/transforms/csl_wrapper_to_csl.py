from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.csl import csl, csl_wrapper
from xdsl.ir import Block, Region, SSAValue
from xdsl.irdl import base
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
from xdsl.utils.isattr import isattr

__DEFAULT_PROG_NAME = "pe_program"
"""
This is the name which will be used by the layout module when calling
`@set_tile_code` if the `csl_wrapper.module` does not provide a `program_name`.
"""


def _collect_params(op: csl_wrapper.ModuleOp) -> list[SSAValue]:
    """
    Creates a list of `csl.param`s which should replace the block arguments in the
    layout and program regions of the wrapper.

    To be called in an `ImplicitBuilder`
    """
    new_args = list[SSAValue]()
    for param in op.params:
        if isa(param.value, builtin.IntegerAttr):
            value = arith.Constant(param.value)
        else:
            value = None
        p = csl.ParamOp(param.key.data, param.type, value)
        new_args.append(SSAValue.get(p))
    return new_args


def _add_to_toplevel(
    rewriter: PatternRewriter,
    op: csl_wrapper.ModuleOp,
    mod: csl.CslModuleOp,
) -> None:
    """Inserts the `csl.module` at the start of the `builtin.module`"""
    assert isa(builtin_module := op.parent_op(), builtin.ModuleOp)
    rewriter.insert_op(mod, InsertPoint.at_start(builtin_module.regions[0].block))


@dataclass(frozen=True)
class RemoveWrapper(RewritePattern):
    """Cleanup pass to remove the now empty csl_wrapper"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, _: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


@dataclass(frozen=True)
class ExtractLayoutModule(RewritePattern):
    """
    Moves the contents of the layout region of the `csl_wrapper.module` into a
    `csl.module`.

    The contents of the region get wrapped in 2 `scf.for` loops.

    `csl_wrapper.yield` is replaced by `csl.set_tile_code`.

    The block args of the block of the region get replaced as follows:
        1: Outer loop counter (x dimension)
        2: Inner loop counter (y dimension)
        3: "width" `csl.param`
        4: "height" `csl.param`
        5..n: Replaced with the `params` of the `csl_wrapper.module` by creating a
              `csl.param` for each of them.
    """

    def add_tile_code(
        self,
        x: SSAValue,
        y: SSAValue,
        width: csl.ParamOp,
        height: csl.ParamOp,
        yield_op: csl_wrapper.YieldOp,
        prog_name: str,
    ) -> tuple[csl.ConstStructOp, csl.SetTileCodeOp]:
        """
        Generate the `csl.set_tile_code` op and the struct needed to call it

        The `csl_wrapper.yield_op` is not modified
        """

        struct = csl.ConstStructOp(
            ("width", width),
            ("height", height),
            *(f for f in yield_op.items()),
        )
        return (
            struct,
            csl.SetTileCodeOp(fname=prog_name, x_coord=x, y_coord=y, params=struct),
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /):
        prog_name = op.program_name.data if op.program_name else __DEFAULT_PROG_NAME
        module_block = Block()

        outer_loop_block = Block()
        outer_loop_block.insert_arg(builtin.IntegerType(16), 0)
        x = outer_loop_block.args[0]

        inner_loop_block = Block()
        inner_loop_block.insert_arg(builtin.IntegerType(16), 0)
        y = inner_loop_block.args[0]

        new_args = list[SSAValue]()

        assert isa(yield_op := op.layout_module.block.last_op, csl_wrapper.YieldOp)
        rewriter.erase_op(yield_op)

        with ImplicitBuilder(module_block):
            const_0 = arith.Constant.from_int_and_width(0, builtin.IntegerType(16))
            const_1 = arith.Constant.from_int_and_width(1, builtin.IntegerType(16))

            const_width = arith.Constant(op.width)
            param_width = csl.ParamOp("width", op.width.type, const_width)

            const_height = arith.Constant(op.height)
            param_height = csl.ParamOp("height", op.height.type, const_height)

            new_args = _collect_params(op)

            layout = csl.LayoutOp(Region())
            with ImplicitBuilder(layout.body.block):
                scf.For(
                    lb=const_0,
                    ub=param_width,
                    step=const_1,
                    iter_args=[],
                    body=outer_loop_block,
                )

                with ImplicitBuilder(outer_loop_block):
                    scf.For(
                        lb=const_0,
                        ub=param_height,
                        step=const_1,
                        iter_args=[],
                        body=inner_loop_block,
                    )
                    scf.Yield()
        rewriter.inline_block(
            op.layout_module.block,
            InsertPoint.at_start(inner_loop_block),
            arg_values=[
                SSAValue.get(x),
                SSAValue.get(y),
                SSAValue.get(param_width),
                SSAValue.get(param_height),
                *new_args,
            ],
        )
        struct, tile_code = self.add_tile_code(
            outer_loop_block.args[0],
            inner_loop_block.args[0],
            param_width,
            param_height,
            yield_op,
            prog_name,
        )
        inner_loop_block.add_ops((struct, tile_code))
        inner_loop_block.add_op(scf.Yield())

        layout_mod = csl.CslModuleOp(
            regions=[Region(module_block)],
            properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.LAYOUT)},
            attributes={"sym_name": builtin.StringAttr(f"{prog_name}_layout")},
        )

        _add_to_toplevel(rewriter, op, layout_mod)


# program: width, height, *params, *layout-yields


@dataclass(frozen=True)
class ExtractProgramModule(RewritePattern):
    """
    Moves the contents of the program region of the `csl_wrapper.module` into a
    `csl.module`.

    The block args of the block of the region get replaced as follows:
        1: Outer loop counter (x dimension)
        2: Inner loop counter (y dimension)
        3: "width" `csl.param`
        4: "height" `csl.param`
        5..n: Replaced with the `params` of the `csl_wrapper.module` by creating a
              `csl.param` for each of them.
    """

    @staticmethod
    def _collect_yield_args(yield_op: csl_wrapper.YieldOp) -> list[csl.ParamOp]:
        params = list[csl.ParamOp]()
        for s, v in yield_op.items():
            assert isattr(ty := v.type, base(csl.ParamOp.T))
            params.append(csl.ParamOp(s, ty))
        return params

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /):
        prog_name = op.program_name.data if op.program_name else __DEFAULT_PROG_NAME
        module_block = Block()
        with ImplicitBuilder(module_block):
            param_width = csl.ParamOp("width", op.width.type)
            param_height = csl.ParamOp("height", op.height.type)

            new_args = _collect_params(op)

        assert isa(yield_op := op.layout_module.block.last_op, csl_wrapper.YieldOp)
        yield_args = self._collect_yield_args(yield_op)
        module_block.add_ops(yield_args)
        assert isa(yield_op := op.program_module.block.last_op, csl_wrapper.YieldOp)
        rewriter.erase_op(yield_op)

        rewriter.inline_block(
            op.program_module.block,
            InsertPoint.at_end(module_block),
            arg_values=[
                SSAValue.get(param_width),
                SSAValue.get(param_height),
                *new_args,
                *(SSAValue.get(y) for y in yield_args),
            ],
        )

        program_module = csl.CslModuleOp(
            regions=[Region(module_block)],
            properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.PROGRAM)},
            attributes={"sym_name": builtin.StringAttr(f"{prog_name}_program")},
        )
        _add_to_toplevel(rewriter, op, program_module)


@dataclass(frozen=True)
class CslWrapperToCslPass(ModulePass):
    """Unwraps the `csl_wrappermodule` into two `csl.module`s."""

    name = "csl-wrapper-to-csl"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        program_module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([ExtractProgramModule()]),
            apply_recursively=False,
        )
        layout_module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([ExtractLayoutModule()]),
            apply_recursively=False,
        )
        cleanup_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([RemoveWrapper()]), apply_recursively=False
        )

        program_module_pass.rewrite_module(op)
        layout_module_pass.rewrite_module(op)
        cleanup_pass.rewrite_module(op)
