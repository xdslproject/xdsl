from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, memref, stencil
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    Signedness,
    StringAttr,
    TensorType,
)
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Attribute, BlockArgument, Operation, OpResult, SSAValue
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
from xdsl.transforms.function_transformations import (
    TIMER_END,
    TIMER_START,
)
from xdsl.utils.hints import isa


def _get_module_wrapper(op: Operation) -> csl_wrapper.ModuleOp | None:
    """
    Return the enclosing csl_wrapper.module
    """
    parent_op = op.parent_op()
    while parent_op:
        if isinstance(parent_op, csl_wrapper.ModuleOp):
            return parent_op
        parent_op = parent_op.parent_op()
    return None


@dataclass(frozen=True)
class ConvertStencilFuncToModuleWrappedPattern(RewritePattern):
    """
    Wraps program in the csl_stencil dialect in a csl_wrapper module.
    Scans the csl_stencil.apply ops for stencil-related params, passing them as properties to the wrapped module
    (note, properties are in return passed as block args to the layout_module and program_module blocks).

    The layout module wrapper can be used to initialise general program module params. This pass generates code
    to initialise stencil-specific program params and yields them from the layout module.
    """

    target: csl.Target
    """
    Specifies the target architecture.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        # erase timer stubs
        if op.is_declaration and op.sym_name.data in [TIMER_START, TIMER_END]:
            rewriter.erase_op(op)
            return
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
        chunk_size: int = 1
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
            if len(shape := apply_op.topo.shape.get_values()) == 2:
                width = max(width, shape[0])
                height = max(height, shape[1])
            else:
                raise ValueError("Stencil accesses must be 2-dimensional at this stage")

            # find max z dimension - we could get this from func args, store ops, or apply ops
            # to support both bufferized and unbufferized csl_stencils, retrieve this from accumulator
            if isinstance(apply_op.done_exchange.block.args[1].type, ShapedType):
                z_dim_no_ghost_cells = max(
                    z_dim_no_ghost_cells,
                    apply_op.done_exchange.block.args[1].type.get_shape()[-1],
                )

            # retrieve z_dim from done_exchange arg[0]
            if stencil.StencilTypeConstr.verifies(
                field_t := apply_op.done_exchange.block.args[0].type
            ) and isa(el_type := field_t.element_type, TensorType | MemRefType):
                # unbufferized csl_stencil
                z_dim = max(z_dim, el_type.get_shape()[-1])
            elif isa(field_t, memref.MemRefType):
                # bufferized csl_stencil
                z_dim = max(z_dim, field_t.get_shape()[-1])

            num_chunks = max(num_chunks, apply_op.num_chunks.value.data)
            if isa(
                buf_t := apply_op.receive_chunk.block.args[0].type,
                TensorType | MemRefType,
            ):
                chunk_size = max(chunk_size, buf_t.get_shape()[-1])

        padded_z_dim: int = chunk_size * num_chunks

        # initialise module op
        module_op = csl_wrapper.ModuleOp(
            width=IntegerAttr(width + (max_distance * 2), 16),
            height=IntegerAttr(height + (max_distance * 2), 16),
            target=self.target,
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

        # set up main function and move func.func ops into this csl.func
        main_func = csl.FuncOp(op.sym_name.data, ((), None))
        func_export = csl.SymbolExportOp(main_func.sym_name, main_func.function_type)
        args_to_ops, arg_mappings = self._translate_function_args(op.args, op.arg_attrs)
        rewriter.inline_block(
            op.body.block,
            InsertPoint.at_start(main_func.body.block),
            arg_mappings,
        )

        # initialise program_module and add main func and empty yield op
        self.initialise_program_module(
            module_op, add_ops=[*args_to_ops, func_export, main_func]
        )

        # replace func.return by unblock_cmd_stream and csl.return
        func_return = main_func.body.block.last_op
        assert isinstance(func_return, func.ReturnOp)
        assert len(func_return.arguments) == 0, (
            "Non-empty returns currently not supported"
        )
        memcpy = module_op.get_program_import("<memcpy/memcpy>")
        unblock_call = csl.MemberCallOp(
            struct=memcpy, fname="unblock_cmd_stream", params=[], result_type=None
        )
        rewriter.replace_op(func_return, [unblock_call, csl.ReturnOp()])

        # replace (now empty) func by module wrapper
        rewriter.replace_op(op, module_op)

    def get_csl_stencil_apply_ops(
        self, op: func.FuncOp
    ) -> Sequence[csl_stencil.ApplyOp]:
        result: list[csl_stencil.ApplyOp] = []
        for apply_op in op.body.walk():
            if isinstance(apply_op, csl_stencil.ApplyOp):
                result.append(apply_op)
        return result

    def _translate_function_args(
        self, args: Sequence[BlockArgument], attrs: ArrayAttr[DictionaryAttr] | None
    ) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
        """
        Args of the top-level function act as the interface to the program and need to
        be translated to writable buffers.
        """
        arg_ops: list[Operation] = []
        arg_op_mapping: list[SSAValue] = []
        ptr_converts: list[Operation] = []
        export_ops: list[Operation] = []
        cast_ops: list[Operation] = []
        import_ops: list[Operation] = []

        if attrs is not None:
            for arg, attr in zip(args, attrs, strict=True):
                assert isinstance(attr, DictionaryAttr)
                if "llvm.name" in attr.data:
                    nh = attr.data["llvm.name"]
                    assert isinstance(nh, StringAttr)
                    arg.name_hint = nh.data

        for arg in args:
            arg_name = arg.name_hint or ("arg" + str(args.index(arg)))

            if isa(arg.type, stencil.FieldType[TensorType[Attribute]]) or isa(
                arg.type, memref.MemRefType
            ):
                arg_t = (
                    csl_stencil_bufferize.tensor_to_memref_type(
                        arg.type.get_element_type()
                    )
                    if isa(arg.type, stencil.FieldType[TensorType[Attribute]])
                    else arg.type
                )
                arg_ops.append(alloc := memref.AllocOp([], [], arg_t))
                ptr_converts.append(
                    address := csl.AddressOfOp(
                        alloc,
                        csl.PtrType.get(
                            arg_t.get_element_type(), is_single=False, is_const=False
                        ),
                    )
                )
                export_ops.append(csl.SymbolExportOp(arg_name, SSAValue.get(address)))
                if arg_t != arg.type:
                    cast_ops.append(
                        cast_op := builtin.UnrealizedConversionCastOp.get(
                            [alloc], [arg.type]
                        )
                    )
                    arg_op_mapping.append(cast_op.outputs[0])
                else:
                    arg_op_mapping.append(alloc.memref)
            # check if this looks like a timer
            elif isinstance(arg.type, llvm.LLVMPointerType) and all(
                isinstance(u.operation, llvm.StoreOp)
                and isinstance(u.operation.value, OpResult)
                and isinstance(u.operation.value.op, func.CallOp)
                and u.operation.value.op.callee.string_value() == TIMER_END
                for u in arg.uses
            ):
                start_end_size = 3
                arg_t = memref.MemRefType(
                    IntegerType(16, Signedness.UNSIGNED), (2 * start_end_size,)
                )
                arg_ops.append(alloc := memref.AllocOp([], [], arg_t))
                ptr_converts.append(
                    address := csl.AddressOfOp(
                        alloc,
                        csl.PtrType.get(
                            arg_t.get_element_type(), is_single=False, is_const=False
                        ),
                    )
                )
                export_ops.append(csl.SymbolExportOp(arg_name, SSAValue.get(address)))
                arg_op_mapping.append(alloc.memref)
                import_ops.append(
                    csl_wrapper.ImportOp(
                        "<time>",
                        field_name_mapping={},
                    )
                )

        return [
            *arg_ops,
            *cast_ops,
            *ptr_converts,
            *export_ops,
            *import_ops,
        ], arg_op_mapping

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
            # import memcpy/get_params and routes
            memcpy = csl_wrapper.ImportOp(
                "<memcpy/get_params>",
                {
                    "width": param_width,
                    "height": param_height,
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
            one = arith.ConstantOp(IntegerAttr(1, 16))
            pattern_minus_one = arith.SubiOp(param_pattern, one)
            width_minus_x = arith.SubiOp(param_width, param_x)
            height_minus_y = arith.SubiOp(param_height, param_y)
            x_lt_pattern_minus_one = arith.CmpiOp(param_x, pattern_minus_one, "slt")
            y_lt_pattern_minus_one = arith.CmpiOp(param_y, pattern_minus_one, "slt")
            width_minus_one_lt_pattern = arith.CmpiOp(
                width_minus_x, param_pattern, "slt"
            )
            height_minus_one_lt_pattern = arith.CmpiOp(
                height_minus_y, param_pattern, "slt"
            )
            or1_op = arith.OrIOp(x_lt_pattern_minus_one, y_lt_pattern_minus_one)
            or2_op = arith.OrIOp(or1_op, width_minus_one_lt_pattern)
            is_border_region_pe = arith.OrIOp(or2_op, height_minus_one_lt_pattern)

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
class LowerTimerFuncCall(RewritePattern):
    """
    Lowers calls to the start and end timer to csl API calls.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.StoreOp, rewriter: PatternRewriter, /):
        if (
            not isinstance(end_call := op.value.owner, func.CallOp)
            or not end_call.callee.string_value() == TIMER_END
            or not (isinstance(start_call := end_call.arguments[0].owner, func.CallOp))
            or not start_call.callee.string_value() == TIMER_START
            or not (wrapper := _get_module_wrapper(op))
            or not isa(op.ptr.type, MemRefType)
        ):
            return

        time_lib = wrapper.get_program_import("<time>")

        three_elem_ptr_type = csl.PtrType(
            memref.MemRefType(op.ptr.type.get_element_type(), (3,)),
            csl.PtrKindAttr(csl.PtrKind.SINGLE),
            csl.PtrConstAttr(csl.PtrConst.VAR),
        )

        rewriter.insert_op(
            [
                three := arith.ConstantOp.from_int_and_width(3, IndexType()),
                load_three := memref.LoadOp.get(op.ptr, [three]),
                addr_of := csl.AddressOfOp(
                    load_three,
                    csl.PtrType.get(
                        op.ptr.type.get_element_type(), is_single=True, is_const=False
                    ),
                ),
                ptrcast := csl.PtrCastOp(addr_of, three_elem_ptr_type),
                csl.MemberCallOp("get_timestamp", None, time_lib, [ptrcast]),
                csl.MemberCallOp("disable_tsc", None, time_lib, []),
            ],
            InsertPoint.before(end_call),
        )
        rewriter.insert_op(
            [
                addr_of := csl.AddressOfOp(
                    op.ptr,
                    csl.PtrType.get(
                        op.ptr.type.get_element_type(), is_single=False, is_const=False
                    ),
                ),
                ptrcast := csl.PtrCastOp(addr_of, three_elem_ptr_type),
                csl.MemberCallOp("enable_tsc", None, time_lib, []),
                csl.MemberCallOp("get_timestamp", None, time_lib, [ptrcast]),
            ],
            InsertPoint.before(start_call),
        )
        rewriter.erase_op(op)
        rewriter.erase_op(end_call)
        rewriter.erase_op(start_call)


@dataclass(frozen=True)
class CslStencilToCslWrapperPass(ModulePass):
    """
    Wraps program in the csl_stencil dialect in a csl_wrapper by translating each
    top-level function to one module wrapper.
    """

    name = "csl-stencil-to-csl-wrapper"

    target: csl.Target
    """
    Specifies the target architecture.
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStencilFuncToModuleWrappedPattern(self.target),
                    LowerTimerFuncCall(),
                ]
            ),
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
