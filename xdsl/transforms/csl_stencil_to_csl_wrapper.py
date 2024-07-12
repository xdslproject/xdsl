from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, stencil
from xdsl.dialects.builtin import IntegerAttr, TensorType
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Attribute
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
class ConvertStencilFuncToModuleWrappedPattern(RewritePattern):
    """
    Wraps program in the csl_stencil dialect in a csl_wrapper module.
    Scans the csl_stencil.apply ops for stencil-related params, passing them as properties to the wrapped module
    (note, properties are in return passed as block args to the layout_module and program_module blocks).

    The layout module wrapper can be used to initialise general program module params. This pass generates code
    to initialise stencil-specific program params and yields them from the layout module.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        # find csl_stencil.apply ops, abort if there are none
        apply_ops = self.get_csl_stencil_apply_ops(op)
        if len(apply_ops) == 0:
            return
        neighbours: int = 1
        width: int = 1
        height: int = 1
        z_dim_no_ghost_cells: int = 1
        z_dim: int = 1
        for apply_op in apply_ops:
            # loop over accesses to get `neighbour` (from which we build `pattern`)
            for ap in apply_op.get_accesses():
                if ap.is_diagonal:
                    raise ValueError("Diagonal accesses are currently not supported")
                # if ap.dims != 2:
                #     raise ValueError("Stencil accesses must be 2-dimensional at this stage")
                if len(ap.offsets) > 0:
                    neighbours = max(
                        neighbours, max(abs(d) for offset in ap.offsets for d in offset)
                    )

            # find max x and y dimensions
            if len(shape := apply_op.topo.shape.data) == 2:
                assert isinstance(
                    shape.data[0].data, int
                ), "Cannot have a float data shape"
                assert isinstance(
                    shape.data[1].data, int
                ), "Cannot have a float data shape"
                width = max(width, shape.data[0].data)
                height = max(height, shape.data[1].data)
            else:
                raise ValueError("Stencil accesses must be 2-dimensional at this stage")

            # find max z dimension - we could get this from func args, store ops, or apply ops
            for result in apply_op.results:
                if isa(result.type, stencil.TempType[TensorType[Attribute]]):
                    z_dim_no_ghost_cells = max(
                        z_dim_no_ghost_cells,
                        result.type.get_element_type().get_shape()[0],
                    )
            for arg in op.args:
                if isa(field_t := arg.type, stencil.FieldType[TensorType[Attribute]]):
                    z_dim = max(z_dim, field_t.get_element_type().get_shape()[0])

        # initialise module op
        module_op = csl_wrapper.ModuleOp(
            width=IntegerAttr(width, 16),
            height=IntegerAttr(height, 16),
            params={
                "z_dim": IntegerAttr(z_dim, 16),
                "pattern": IntegerAttr(neighbours + 1, 16),
            },
        )

        self.initialise_layout_module(module_op)
        module_op.update_program_block_args_from_layout()
        module_op.set_program_name(op.sym_name)

        # add func args to program_module block args
        for block_arg in op.body.block.args:
            module_op.program_module.block.insert_arg(
                block_arg.type, len(module_op.program_module.block.args)
            )

        # replace func.return
        func_return = op.body.block.last_op
        assert isinstance(func_return, func.Return)
        assert (
            len(func_return.arguments) == 0
        ), "Non-empty returns currently not supported"
        rewriter.replace_op(func_return, csl.ReturnOp())

        # set up main function and move func.func ops into this csl.func
        main_func = csl.FuncOp(op.sym_name.data, ((), None))
        rewriter.inline_block(
            op.body.block,
            InsertPoint.at_start(main_func.body.block),
            module_op.exported_symbols,
        )

        # add main and empty yield to program_module
        module_op.program_module.block.add_ops([main_func, csl_wrapper.YieldOp([], [])])

        # replace (now empty) func by module wrapper
        rewriter.replace_matched_op(module_op)

    def get_csl_stencil_apply_ops(
        self, op: func.FuncOp
    ) -> Sequence[csl_stencil.ApplyOp]:
        result: list[csl_stencil.ApplyOp] = []
        for apply_op in op.body.walk():
            if isinstance(apply_op, csl_stencil.ApplyOp):
                result.append(apply_op)
        return result

    def initialise_layout_module(self, module_op: csl_wrapper.ModuleOp):
        """Initialises the layout_module (wrapper block) by setting up (esp. stencil-related) program params"""

        # extract layout module params as the function has linear complexity
        param_width = module_op.get_layout_param("width")
        param_height = module_op.get_layout_param("height")
        param_x = module_op.get_layout_param("x")
        param_y = module_op.get_layout_param("y")
        param_pattern = module_op.get_layout_param("pattern")

        # fill layout module wrapper block with ops
        with ImplicitBuilder(module_op.layout_module.block):
            # set up LAUNCH
            zero = arith.Constant(IntegerAttr(0, 16))
            launch = csl.GetColorOp(zero)

            # import memcpy/get_params and routes
            memcpy = csl_wrapper.ImportModuleOp(
                "<memcpy/get_params>",
                {
                    "width": param_width,
                    "height": param_height,
                    "LAUNCH": launch.res,
                },
            )
            routes = csl_wrapper.ImportModuleOp(
                "routes.csl",
                {
                    "pattern": param_pattern,
                    "peWidth": param_width,
                    "peHeight": param_height,
                },
            )

            # set up program param `stencil_comms_params`
            all_routes = csl.MemberCallOp(
                "computeAllRoutes",
                csl.ComptimeStructType(),
                routes,
                params=[
                    param_x,
                    param_y,
                    param_width,
                    param_height,
                    param_pattern,
                ],
            )
            # set up program param `memcpy_params`
            memcpy_params = csl.MemberCallOp(
                "get_params",
                csl.ComptimeStructType(),
                memcpy,
                params=[
                    param_x,
                ],
            )

            # set up program param `is_border_region_pe`
            one = arith.Constant(IntegerAttr(1, 16))
            pattern_minus_one = arith.Subi(one, param_pattern)
            width_minus_x = arith.Subi(param_width, param_x)
            height_minus_y = arith.Subi(param_height, param_y)
            x_lt_pattern_minus_one = arith.Cmpi(param_x, pattern_minus_one, "slt")
            y_lt_pattern_minus_one = arith.Cmpi(param_y, pattern_minus_one, "slt")
            width_minus_one_lt_pattern = arith.Cmpi(width_minus_x, param_pattern, "slt")
            height_minus_one_lt_pattern = arith.Cmpi(
                height_minus_y, param_pattern, "slt"
            )
            or1_op = arith.OrI(x_lt_pattern_minus_one, y_lt_pattern_minus_one)
            or2_op = arith.OrI(or1_op, width_minus_one_lt_pattern)
            is_border_region_pe = arith.OrI(or2_op, height_minus_one_lt_pattern)

            # yield things as named params to the program module
            csl_wrapper.YieldOp.from_field_name_mapping(
                field_name_mapping={
                    "memcpy_params": memcpy_params.results[0],
                    "stencil_comms_params": all_routes.results[0],
                    "isBorderRegionPE": is_border_region_pe.result,
                }
            )


@dataclass(frozen=True)
class CslStencilToCslWrapperPass(ModulePass):
    """
    Wraps program in the csl_stencil dialect in a csl_wrapper by translating each
    top-level function to one module wrapper.
    """

    name = "csl-stencil-to-csl-wrapper"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStencilFuncToModuleWrappedPattern(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
