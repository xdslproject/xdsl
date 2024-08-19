from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, memref, stencil
from xdsl.dialects.builtin import IntegerAttr, StringAttr, TensorType, UnitAttr
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms import csl_stencil_bufferize
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
        max_distance: int = 1
        width: int = 1
        height: int = 1
        z_dim_no_ghost_cells: int = 1
        z_dim: int = 1
        num_chunks: int = 1
        for apply_op in apply_ops:
            # loop over accesses to get max_distance (from which we build `pattern`)
            for ap in apply_op.get_accesses():
                if ap.is_diagonal:
                    raise ValueError("Diagonal accesses are currently not supported")
                if len(ap.offsets) > 0:
                    if ap.dims != 2:
                        raise ValueError(
                            "Stencil accesses must be 2-dimensional at this stage"
                        )
                    max_distance = max(max_distance, ap.max_distance())

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

            num_chunks = max(num_chunks, apply_op.num_chunks.value.data)

        # some computations we don't need to do in CSL
        chunk_size: int = (z_dim // num_chunks) + (0 if z_dim % num_chunks == 0 else 1)
        padded_z_dim: int = chunk_size * num_chunks

        # initialise module op
        module_op = csl_wrapper.ModuleOp(
            width=IntegerAttr(width, 16),
            height=IntegerAttr(height, 16),
            params={
                "z_dim": IntegerAttr(z_dim, 16),
                "pattern": IntegerAttr(max_distance + 1, 16),
                "num_chunks": IntegerAttr(num_chunks, 16),
                "chunk_size": IntegerAttr(chunk_size, 16),
                "padded_z_dim": IntegerAttr(padded_z_dim, 16),
            },
        )

        self.initialise_layout_module(module_op)
        module_op.program_name = op.sym_name

        # add yield op args to program_module block args
        module_op.update_program_block_args()

        # replace func.return
        func_return = op.body.block.last_op
        assert isinstance(func_return, func.Return)
        assert (
            len(func_return.arguments) == 0
        ), "Non-empty returns currently not supported"
        rewriter.replace_op(func_return, csl.ReturnOp())

        # set up main function and move func.func ops into this csl.func
        main_func = csl.FuncOp(op.sym_name.data, ((), None))
        args_to_ops, arg_mappings = self._translate_function_args(op.args)
        rewriter.inline_block(
            op.body.block,
            InsertPoint.at_start(main_func.body.block),
            arg_mappings,
        )

        # initialise program_module and add main func and empty yield op
        self.initialise_program_module(module_op, add_ops=[*args_to_ops, main_func])

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

    def _translate_function_args(
        self, args: Sequence[BlockArgument]
    ) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
        """
        Args of the top-level function act as the interface to the program and need to be translated to writable buffers.
        """
        arg_ops: list[Operation] = []
        arg_op_mapping: list[SSAValue] = []
        for arg in args:
            arg_name = arg.name_hint or ("arg" + str(args.index(arg)))

            if isa(arg.type, stencil.FieldType[TensorType[Attribute]]):
                arg_t = csl_stencil_bufferize.tensor_to_memref_type(
                    arg.type.get_element_type()
                )
                arg_ops.append(
                    memref.Global.get(
                        sym_name=StringAttr(arg_name),
                        sym_type=csl_stencil_bufferize.tensor_to_memref_type(
                            arg.type.get_element_type()
                        ),
                        initial_value=UnitAttr(),
                        sym_visibility=StringAttr("public"),
                    )
                )
                arg_ops.append(get_global_op := memref.GetGlobal(arg_name, arg_t))
                arg_ops.append(
                    cast_op := builtin.UnrealizedConversionCastOp.get(
                        [get_global_op], [arg.type]
                    )
                )
                arg_op_mapping.append(cast_op.outputs[0])
            elif isa(arg.type, memref.MemRefType[Attribute]):
                arg_ops.append(
                    memref.Global.get(
                        sym_name=StringAttr(arg_name),
                        sym_type=arg.type,
                        initial_value=UnitAttr(),
                        sym_visibility=StringAttr("public"),
                    )
                )
                arg_ops.append(get_global_op := memref.GetGlobal(arg_name, arg.type))
                arg_op_mapping.append(get_global_op.memref)

        return arg_ops, arg_op_mapping

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
            memcpy = csl_wrapper.ImportOp(
                "<memcpy/get_params>",
                {
                    "width": param_width,
                    "height": param_height,
                    "LAUNCH": launch.res,
                },
            )
            routes = csl_wrapper.ImportOp(
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

    def initialise_program_module(
        self, module_op: csl_wrapper.ModuleOp, add_ops: Sequence[Operation]
    ):
        with ImplicitBuilder(module_op.program_module.block):
            csl_wrapper.ImportOp(
                "<memcpy/memcpy>",
                field_name_mapping={"": module_op.get_program_param("memcpy_params")},
            )
            csl_wrapper.ImportOp(
                "stencil_comms.csl",
                field_name_mapping={
                    "pattern": module_op.get_program_param("pattern"),
                    "chunkSize": module_op.get_program_param("chunk_size"),
                    "": module_op.get_program_param("stencil_comms_params"),
                },
            )
        module_op.program_module.block.add_ops(add_ops)
        module_op.program_module.block.add_op(csl_wrapper.YieldOp([], []))


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
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
