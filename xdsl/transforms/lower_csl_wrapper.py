from dataclasses import dataclass
from pathlib import Path

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.csl import csl, csl_wrapper
from xdsl.ir import Block, Operation, Region, SSAValue
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

_DEFAULT_PROG_NAME = "pe_program"
"""
This is the name which will be used by the layout module when calling
`@set_tile_code` if the `csl_wrapper.module` does not provide a `program_name`.
"""


@dataclass(frozen=True)
class ExtractCslModules(RewritePattern):
    params_as_consts: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /):
        program_module = self.lower_program_module(op, rewriter)
        layout_module = self.lower_layout_module(op, rewriter)
        rewriter.replace_op(op, [layout_module, program_module])

    def _collect_params(
        self, op: csl_wrapper.ModuleOp
    ) -> tuple[SSAValue, SSAValue, list[SSAValue]]:
        """
        Creates a list of `csl.param`s which should replace the block arguments in the
        layout and program regions of the wrapper.

        To be called in an `ImplicitBuilder`

        Returns width, height, and a list of other params as SSAValues.
        Params can alternatively be lowered to constants via the `params_as_consts` flag.
        """
        width = arith.ConstantOp(op.width).result
        height = arith.ConstantOp(op.height).result
        if not self.params_as_consts:
            width = csl.ParamOp("width", op.width.type, width).res
            height = csl.ParamOp("height", op.height.type, height).res

        params = list[SSAValue]()
        for param in op.params:
            if isinstance(param.value, builtin.IntegerAttr):
                value = arith.ConstantOp(param.value)
            else:
                value = None
            if value and self.params_as_consts:
                params.append(value.result)
            else:
                p = csl.ParamOp(param.key.data, param.type, value)
                params.append(p.res)
        return width, height, params

    def add_tile_code(
        self,
        x: SSAValue,
        y: SSAValue,
        width: SSAValue,
        height: SSAValue,
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
            csl.SetTileCodeOp(
                fname=f"{prog_name}.csl", x_coord=x, y_coord=y, params=struct
            ),
        )

    def lower_layout_module(
        self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /
    ) -> csl.CslModuleOp:
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

        prog_name = op.program_name.data if op.program_name else _DEFAULT_PROG_NAME
        module_block = Block()

        outer_loop_block = Block()
        outer_loop_block.insert_arg(builtin.IntegerType(16), 0)
        x = outer_loop_block.args[0]
        x.name_hint = "x"

        inner_loop_block = Block()
        inner_loop_block.insert_arg(builtin.IntegerType(16), 0)
        y = inner_loop_block.args[0]
        y.name_hint = "y"

        assert isa(yield_op := op.layout_module.block.last_op, csl_wrapper.YieldOp)
        rewriter.erase_op(yield_op)

        with ImplicitBuilder(module_block):
            const_0 = arith.ConstantOp.from_int_and_width(0, builtin.IntegerType(16))
            const_1 = arith.ConstantOp.from_int_and_width(1, builtin.IntegerType(16))

            param_width, param_height, params_from_block_args = self._collect_params(op)

            layout = csl.LayoutOp(Region())
            with ImplicitBuilder(layout.body.block):
                csl.SetRectangleOp(operands=[param_width, param_height])
                scf.ForOp(
                    lb=const_0,
                    ub=param_width,
                    step=const_1,
                    iter_args=[],
                    body=outer_loop_block,
                )

                with ImplicitBuilder(outer_loop_block):
                    scf.ForOp(
                        lb=const_0,
                        ub=param_height,
                        step=const_1,
                        iter_args=[],
                        body=inner_loop_block,
                    )
                    scf.YieldOp()
        rewriter.inline_block(
            op.layout_module.block,
            InsertPoint.at_start(inner_loop_block),
            arg_values=[
                SSAValue.get(x),
                SSAValue.get(y),
                param_width,
                param_height,
                *params_from_block_args,
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
        inner_loop_block.add_op(scf.YieldOp())

        layout_mod = csl.CslModuleOp(
            regions=[Region(module_block)],
            properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.LAYOUT)},
            attributes={"sym_name": builtin.StringAttr(f"{prog_name}_layout")},
        )
        return layout_mod

    @staticmethod
    def _collect_yield_args(yield_op: csl_wrapper.YieldOp) -> list[csl.ParamOp]:
        params = list[csl.ParamOp]()
        for s, v in yield_op.items():
            assert csl.ParamOpAttrConstr.verifies(ty := v.type)
            params.append(csl.ParamOp(s, ty))
        return params

    def lower_program_module(
        self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /
    ) -> csl.CslModuleOp:
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

        prog_name = op.program_name.data if op.program_name else _DEFAULT_PROG_NAME
        module_block = Block()
        with ImplicitBuilder(module_block):
            param_width, param_height, params_from_block_args = self._collect_params(op)

        assert isa(yield_op := op.layout_module.block.last_op, csl_wrapper.YieldOp)
        yield_args = self._collect_yield_args(yield_op)
        module_block.add_ops(yield_args)
        assert isa(yield_op := op.program_module.block.last_op, csl_wrapper.YieldOp)
        rewriter.erase_op(yield_op)

        rewriter.inline_block(
            op.program_module.block,
            InsertPoint.at_end(module_block),
            arg_values=[
                param_width,
                param_height,
                *params_from_block_args,
                *(y.res for y in yield_args),
            ],
        )

        program_module = csl.CslModuleOp(
            regions=[Region(module_block)],
            properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.PROGRAM)},
            attributes={"sym_name": builtin.StringAttr(f"{prog_name}_program")},
        )
        return program_module


@dataclass(frozen=True)
class LowerImport(RewritePattern):
    """
    Replace the `csl_wrapper.import` with the equivalent `csl.import`.

    Hoist the import and all ops it depends on to the module scope (as is required by CSL)
    """

    def _get_csl_mod(self, op: Operation) -> csl.CslModuleOp:
        """
        Find the parent `csl.module` of the current op
        """

        if isinstance(op, csl.CslModuleOp):
            return op
        assert (parent := op.parent_op()) is not None
        return self._get_csl_mod(parent)

    def _collect_ops(self, op: Operation, ops: list[Operation]) -> list[Operation]:
        """
        Detach the op from its current location and store it in the list

        Do this recursively for all operands of each operation.

        NOTE: This op's dependencies are added to the list first to preserve
              the order in which they get added back to the module.
        """
        op.detach()
        for operand in op.operands:
            owner = operand.owner
            assert isinstance(owner, Operation)
            self._collect_ops(owner, ops)
        ops.append(op)
        return ops

    def _make_import_struct(self, import_op: csl_wrapper.ImportOp):
        """
        Create the struct to be passed to `@import_module`

        Handles the case where multiple structs need to be concatinated before
        being imported (this is indicated by an empty field name in
        `csl_wrapper.import`). All required intermediate structs get returned
        as a list.

        Last struct in the list is to be used in `csl.import`
        """

        fields = list[tuple[str, SSAValue | Operation]]()
        structs_to_concat = list[SSAValue | Operation]()
        for fname, op in zip(import_op.fields, import_op.ops):
            if fname.data != "":
                fields.append((fname.data, op))
            else:
                structs_to_concat.append(op)
        out_structs: list[Operation] = [csl.ConstStructOp(*fields)]
        for s in structs_to_concat:
            out_structs.append(csl.ConcatStructOp(out_structs[-1], s))
        return out_structs

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ImportOp, rewriter: PatternRewriter, /):
        csl_mod = self._get_csl_mod(op)
        ops = self._collect_ops(op, [])
        structs = self._make_import_struct(op)
        import_ = csl.ImportModuleConstOp(
            op.module, structs[-1] if len(structs) > 0 else None
        )
        import_.result.name_hint = Path(op.module.data.strip("<>")).stem

        rewriter.insert_op(ops, InsertPoint.at_start(csl_mod.body.block))
        rewriter.insert_op(structs, InsertPoint.before(op))
        rewriter.replace_op(op, import_)


@dataclass(frozen=True)
class LowerCslWrapperPass(ModulePass):
    """Unwraps the `csl_wrappermodule` into two `csl.module`s."""

    name = "lower-csl-wrapper"

    params_as_consts: bool = False
    """
    Set to lower numerical module wrapper params that have a default value to constants,
    instead of lowering to csl params. Set flag to disallow command line overrides.
    Module wrapper params without a default value will always be lowered to csl params
    (hint: consider removing default values in cases where this is desired).
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ExtractCslModules(params_as_consts=self.params_as_consts),
                    LowerImport(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
