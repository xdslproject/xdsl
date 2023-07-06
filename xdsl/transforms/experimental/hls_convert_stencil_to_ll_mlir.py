from dataclasses import dataclass
from typing import Literal, TypeVar, Iterable, cast

from warnings import warn

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter
from xdsl.ir import Block, MLContext, Region, Operation, SSAValue
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import FunctionType, i32, IntegerType, f64
from xdsl.dialects.func import FuncOp, Call, Return
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin, gpu, llvm
from xdsl.dialects.arith import Constant

from xdsl.dialects.stencil import CastOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
    FieldType,
    LoadOp,
    ReturnOp,
    StencilBoundsAttr,
    StencilType,
    StoreOp,
    TempType,
    ExternalLoadOp,
    ExternalStoreOp,
    IndexOp,
)

from xdsl.dialects.experimental.hls import (
    HLSStream,
    HLSStreamType,
    # HLSExternalLoadOp
)

from xdsl.passes import ModulePass

from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

from xdsl.builder import Builder

from xdsl.dialects.llvm import LLVMPointerType, LLVMStructType, LLVMArrayType

from xdsl.ir.core import BlockArgument

_TypeElement = TypeVar("_TypeElement", bound=Attribute)

# TODO docstrings and comments


def StencilToMemRefType(
    input_type: StencilType[_TypeElement],
) -> MemRefType[_TypeElement]:
    return MemRefType.from_element_type_and_shape(
        input_type.element_type, input_type.get_shape()
    )


@dataclass
class CastOpToMemref(RewritePattern):
    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        assert isa(op.result.typ, FieldType[Attribute])
        assert isinstance(op.result.typ.bounds, StencilBoundsAttr)

        result_typ = StencilToMemRefType(op.result.typ)

        cast = memref.Cast.get(op.field, result_typ)

        if self.target == "gpu":
            unranked = memref.Cast.get(
                cast.dest,
                memref.UnrankedMemrefType.from_type(op.result.typ.element_type),
            )
            register = gpu.HostRegisterOp(unranked.dest)
            rewriter.insert_op_after_matched_op([unranked, register])
        rewriter.replace_matched_op(cast)


# Collect up to 'number' block arguments by walking up the region tree
# and collecting block arguments as we reach new parent regions.
def collectBlockArguments(number: int, block: Block):
    args = []

    while len(args) < number:
        args = list(block.args[0 : number - len(args)]) + args

        parent = block.parent_block()
        if parent is None:
            break

        block = parent

    return args


def update_return_target(
    return_targets: dict[ReturnOp, list[SSAValue | None]],
    old_target: SSAValue,
    new_target: SSAValue,
):
    for targets in return_targets.values():
        for i, target in enumerate(targets):
            if target == old_target:
                targets[i] = new_target


@dataclass
class ReturnOpToMemref(RewritePattern):
    return_target: dict[ReturnOp, list[SSAValue | None]]

    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        store_list: list[memref.Store] = []

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp | scf.For)

        for j in range(len(op.arg)):
            target = self.return_target[op][j]

            if target is None:
                break

            assert isinstance(target.typ, builtin.ShapedType)

            assert (block := op.parent_block()) is not None

            dims = target.typ.get_num_dims()

            args = collectBlockArguments(dims, block)

            if self.target == "gpu":
                args = list(reversed(args))

            store_list.append(memref.Store.get(op.arg[j], target, args))

        rewriter.replace_matched_op([*store_list])


def assert_subset(field: FieldType[Attribute], temp: TempType[Attribute]):
    assert isinstance(field.bounds, StencilBoundsAttr)
    assert isinstance(temp.bounds, StencilBoundsAttr)
    if temp.bounds.lb < field.bounds.lb or temp.bounds.ub > field.bounds.ub:
        raise VerifyException(
            "The stencil computation requires a field with lower bound at least "
            f"{temp.bounds.lb}, got {field.bounds.lb}, min: {min(field.bounds.lb, temp.bounds.lb)}"
        )


class IndexOpToLoopSSA(RewritePattern):
    @staticmethod
    def discover_enclosing_loops(op: Operation) -> Iterable[scf.For | scf.ParallelOp]:
        parent_op = op.parent_op()
        if parent_op is not None:
            yield from IndexOpToLoopSSA.discover_enclosing_loops(parent_op)
        if isa(op, scf.For) or isa(op, scf.ParallelOp):
            yield op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IndexOp, rewriter: PatternRewriter, /):
        # We do not currently support an offset in indexop, therefore check
        # that this is all set to zero as otherwise it will not be handled
        for offset in op.offset:
            assert offset == 0
        enclosing_loops = list(IndexOpToLoopSSA.discover_enclosing_loops(op))
        # The first block argument is the loop iterator
        loop_op = enclosing_loops[op.dim.value.data]
        assert isa(loop_op, scf.For) or isa(loop_op, scf.ParallelOp)
        assert len(loop_op.body.blocks) == 1
        assert len(loop_op.body.block.args) >= 1
        replacement_ssa = loop_op.body.block.args[0]
        op.results[0].replace_by(replacement_ssa)
        rewriter.erase_op(op)


class LoadOpToMemref(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        field = op.field.typ
        assert isa(field, FieldType[Attribute])
        assert isa(field.bounds, StencilBoundsAttr)
        temp = op.res.typ
        assert isa(temp, TempType[Attribute])
        assert isa(temp.bounds, StencilBoundsAttr)

        assert_subset(field, temp)

        offsets = [i for i in -field.bounds.lb]
        sizes = [i for i in temp.get_shape()]
        strides = [1] * len(sizes)

        subview = memref.Subview.from_static_parameters(
            op.field, StencilToMemRefType(field), offsets, sizes, strides
        )

        rewriter.replace_matched_op(subview)
        name = None
        if subview.source.name_hint:
            name = subview.source.name_hint + "_loadview"
        subview.result.name_hint = name


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter):
    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.block

    for idx, arg in enumerate(entry.args):
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.replace_operand(use.index, op.args[idx])
        entry.erase_arg(arg)

    rewriter.insert_block_argument(entry, 0, builtin.IndexType())

    return rewriter.move_region_contents_to_new_regions(op.region)


@dataclass
class ApplyOpToParallel(RewritePattern):
    return_targets: dict[ReturnOp, list[SSAValue | None]]

    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        res_typ = op.res[0].typ
        assert isa(res_typ, TempType[Attribute])
        assert isinstance(res_typ.bounds, StencilBoundsAttr)

        # Get this apply's ReturnOp
        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))

        body = prepare_apply_body(op, rewriter)
        body.block.add_op(scf.Yield.get())
        dim = res_typ.get_num_dims()

        # Then create the corresponding scf.parallel
        boilerplate_ops = [
            *(
                lowerBounds := [
                    arith.Constant.from_int_and_width(x, builtin.IndexType())
                    for x in res_typ.bounds.lb
                ]
            ),
            one := arith.Constant.from_int_and_width(1, builtin.IndexType()),
            *(
                upperBounds := [
                    arith.Constant.from_int_and_width(x, builtin.IndexType())
                    for x in res_typ.bounds.ub
                ]
            ),
        ]

        # Generate an outer parallel loop as well as two inner sequential
        # loops. The inner sequential loops ensure that the computational
        # kernel itself is not slowed down by the OpenMP runtime.
        match self.target:
            case "cpu":
                current_region = body
                for i in range(1, dim):
                    for_op = scf.For.get(
                        lb=lowerBounds[-i],
                        ub=upperBounds[-i],
                        step=one,
                        iter_args=[],
                        body=current_region,
                    )
                    block = Block(
                        ops=[for_op, scf.Yield.get()], arg_types=[builtin.IndexType()]
                    )
                    current_region = Region(block)

                p = scf.ParallelOp.get(
                    lowerBounds=[lowerBounds[0]],
                    upperBounds=[upperBounds[0]],
                    steps=[one],
                    body=current_region,
                )
            case "gpu":
                stencil_rank = len(upperBounds)
                boilerplate_ops.insert(
                    1, zero := arith.Constant.from_int_and_width(0, builtin.IndexType())
                )
                p = scf.ParallelOp.get(
                    lowerBounds=list(reversed(lowerBounds))
                    + [zero] * (3 - stencil_rank),
                    upperBounds=list(reversed(upperBounds))
                    + [one] * (3 - stencil_rank),
                    steps=[one] * 3,
                    body=body,
                )
                for _ in range(3 - 1):
                    rewriter.insert_block_argument(p.body.block, 0, builtin.IndexType())

        # Handle returnd values
        for result in op.res:
            assert isa(
                result.typ, TempType[Attribute]
            ), f"Expected return value to be a !{TempType.name}"
            assert isinstance(
                result.typ.bounds, StencilBoundsAttr
            ), f"Expected output to be sized before lowering. {result.typ}"
            shape = result.typ.get_shape()
            element_type = result.typ.element_type

            # If it is buffered, allocate the buffer
            if any(isinstance(use.operation, BufferOp) for use in result.uses):
                alloc = memref.Alloc.get(element_type, shape=shape)
                alloc_type = alloc.memref.typ
                assert isa(alloc_type, MemRefType[Attribute])

                offset = list(-result.typ.bounds.lb)

                view = memref.Subview.from_static_parameters(
                    alloc,
                    alloc_type,
                    offset,
                    shape,
                    [1] * result.typ.get_num_dims(),
                )
                rewriter.insert_op_before_matched_op((alloc, view))
                update_return_target(self.return_targets, result, view.result)

        deallocs: list[Operation] = []
        # Handle input buffer deallocation
        for input in op.args:
            # Is this input a temp buffer?
            if isinstance(input.typ, TempType) and isinstance(input.owner, BufferOp):
                block = op.parent_block()
                assert block is not None
                self_index = block.get_operation_index(op)
                # Is it its last use?
                if not any(
                    use.operation.parent_block() is block
                    and block.get_operation_index(use.operation) > self_index
                    for use in input.uses
                ):
                    # Then deallocate it
                    deallocs.append(memref.Dealloc.get(input))

        # Get the maybe updated results
        new_results: list[SSAValue | None] = []
        new_results = self.return_targets[return_op]
        # Replace with the loop and necessary constants.
        rewriter.insert_op_before_matched_op([*boilerplate_ops, p])
        rewriter.insert_op_after_matched_op([*deallocs])
        rewriter.replace_matched_op([], new_results)


@dataclass
class AccessOpToMemref(RewritePattern):
    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        temp = op.temp.typ
        assert isa(temp, TempType[Attribute])
        assert isinstance(temp.bounds, StencilBoundsAttr)

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        memref_offset = op.offset
        off_const_ops = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in memref_offset
        ]

        args = collectBlockArguments(len(memref_offset), block)
        if self.target == "gpu":
            args = reversed(args)

        off_sum_ops = [arith.Addi(i, x) for i, x in zip(args, off_const_ops)]

        load = memref.Load.get(op.temp, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load], [load.res])


class StencilTypeConversionFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = []
        for arg in op.body.block.args:
            if isa(arg.typ, FieldType[Attribute]):
                memreftyp = StencilToMemRefType(arg.typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)
            else:
                inputs.append(arg.typ)
        outputs: list[Attribute] = [
            StencilToMemRefType(out) if isa(out, FieldType[Attribute]) else out
            for out in op.function_type.outputs
        ]
        op.attributes["function_type"] = FunctionType.from_lists(inputs, outputs)


class UpdateLoopCarriedVarTypes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        for i in range(len(op.iter_args)):
            block_arg = op.body.block.args[i + 1]
            iter_typ = op.iter_args[i].typ
            if block_arg.typ != iter_typ:
                rewriter.modify_block_argument_type(block_arg, iter_typ)
            y = cast(scf.Yield, op.body.ops.last)
            y.arguments[i].typ = iter_typ
            if op.res[i].typ != iter_typ:
                op.res[i].typ = iter_typ


@dataclass
class StencilStoreToSubview(RewritePattern):
    return_targets: dict[ReturnOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        stores = [o for o in op.walk() if isinstance(o, StoreOp)]

        for store in stores:
            field = store.field
            assert isa(field.typ, FieldType[Attribute])
            assert isa(field.typ.bounds, StencilBoundsAttr)
            temp = store.temp
            assert isa(temp.typ, TempType[Attribute])
            offsets = [i for i in -field.typ.bounds.lb]
            sizes = [i for i in temp.typ.get_shape()]
            subview = memref.Subview.from_static_parameters(
                field,
                StencilToMemRefType(field.typ),
                offsets,
                sizes,
                [1] * len(sizes),
            )
            name = None
            if subview.source.name_hint:
                name = subview.source.name_hint + "_storeview"
            subview.result.name_hint = name
            if isinstance(field.owner, Operation):
                rewriter.insert_op_after(subview, field.owner)
            else:
                rewriter.insert_op_at_start(subview, field.owner)

            rewriter.erase_op(store)

            update_return_target(self.return_targets, field, subview.result)


class BufferOpCleanUp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op([], [op.temp])


class TrivialExternalLoadOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        assert isa(op.result.typ, FieldType[Attribute])
        op.result.typ = StencilToMemRefType(op.result.typ)

        if op.field.typ == op.result.typ:
            rewriter.replace_matched_op([], [op.field])
        pass


class TrivialExternalStoreOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


def return_target_analysis(module: builtin.ModuleOp):
    return_targets: dict[ReturnOp, list[SSAValue | None]] = {}

    for op in module.walk():
        if not isinstance(op, ReturnOp):
            continue

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        return_targets[op] = []
        for res in list(apply.res):
            store = [
                use.operation
                for use in list(res.uses)
                if isinstance(use.operation, StoreOp | BufferOp)
            ]

            if len(store) > 1:
                warn("Each stencil result should be stored only once.")
                continue

            elif len(store) == 0:
                field = None
            elif isinstance(store[0], StoreOp):
                field = store[0].field
            # then it's a BufferOp
            else:
                field = store[0].temp

            return_targets[op].append(field)

    return return_targets


@dataclass
class FuncArgsToLLVMPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        for arg in op.args:
            print("----> ARG: ", arg)
            if isa(arg.typ, FieldType[Attribute]):
                print("HOLA")
                print("-----> uses: ")
                for use in arg.uses:
                    print("-----> ", use.operation)
                    print("-----> ", use.operation.operands)
                    used = use.operation.operands[use.index]
                    print("-----> (1) operand[index]: ", used)
                    field_type = used.typ.get_element_type()
                    rewriter.modify_block_argument_type(
                        used, llvm.LLVMPointerType.typed(field_type)
                    )
                    print("-----> (2) operand[index]: ", used)
                print(arg.typ.get_element_type())
                field_type = arg.typ.get_element_type()
                rewriter.modify_block_argument_type(
                    arg, llvm.LLVMPointerType.typed(field_type)
                )


@dataclass
class ExtractForBody(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: Rewriter, /):
        # Call to the function with body of the for loop
        stencil_func_args: list[SSAValue] = []
        stencil_func_args_typ: list[Attribute] = []
        stencil_func_args.append(op.step)
        stencil_func_args_typ.append(op.step.typ)
        for arg in op.iter_args:
            stencil_func_args.append(arg)
            stencil_func_args_typ.append(arg.typ)

        stencil_func_res = []
        stencil_func_res_typ = []
        for r in op.res:
            stencil_func_res.append(r)
            stencil_func_res_typ.append(r.typ)

        stencil_func_rettype = op.parent_op().function_type.outputs
        call = Call.get("stencil_function", stencil_func_args, stencil_func_res_typ)

        # Definition of stencil_func from the body of the foor loop
        for_region = rewriter.move_region_contents_to_new_regions(op.body)
        stencil_func = FuncOp.from_region(
            "stencil_function", stencil_func_args_typ, stencil_func_res_typ, for_region
        )

        yield_op = stencil_func.body.block.last_op
        ret_op = Return(*yield_op.arguments)

        rewriter.replace_op(yield_op, ret_op)

        self.module.body.block.add_op(stencil_func)
        rewriter.replace_op(op, call, None)


@dataclass
class StencilExternalLoadToHLSExternalLoad(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.set_load_data_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: Rewriter, /):
        field = op.field
        res = op.result

        # Find the llvm.ptr to external memory that genrates the argument to the stencil.external_load. For PSyclone, this is
        # an argument to the parent function. TODO: this might need to be tested and generalised for other codes. Also, we are
        # considering that the function argument will be the second to insertvalue, but we're walking up trhough the second to
        # avoid bumping into arith.constants (see the mlir ssa).
        new_op = field
        func_arg = None

        while not isa(func_arg, BlockArgument):
            func_arg = new_op.owner.operands[-1]
            new_op = new_op.owner.operands[0]

        func_arg_type = func_arg.typ.type

        func_arg_elem_type = func_arg.typ.type

        stencil_type = LLVMStructType.from_type_list(
            [
                LLVMArrayType.from_size_and_type(
                    3,
                    LLVMArrayType.from_size_and_type(
                        3, LLVMArrayType.from_size_and_type(3, f64)
                    ),
                )
            ]
        )

        struct_type = LLVMStructType.from_type_list([func_arg_elem_type])
        struct_stencil_type = LLVMStructType.from_type_list([stencil_type])

        shape = field.typ.get_shape()
        shape_x = Constant.from_int_and_width(shape[0], i32)
        shape_y = Constant.from_int_and_width(shape[1], i32)
        shape_z = Constant.from_int_and_width(shape[2], i32)
        data_stream = HLSStream.get(struct_type)
        stencil_stream = HLSStream.get(struct_stencil_type)

        threedload_call = Call.get(
            "load_data", [func_arg, data_stream, shape_x, shape_y, shape_z], []
        )

        shift_buffer_call = Call.get(
            "shift_buffer", [data_stream, stencil_stream, shape_x, shape_y, shape_z], []
        )

        rewriter.insert_op_before_matched_op(
            [
                data_stream,
                stencil_stream,
                shape_x,
                shape_y,
                shape_z,
                threedload_call,
                shift_buffer_call,
            ]
        )

        if not self.set_load_data_declaration:
            load_data_func = FuncOp.external(
                "load_data",
                [
                    func_arg.typ,
                    LLVMPointerType.typed(data_stream.elem_type),
                    i32,
                    i32,
                    i32,
                ],
                [],
            )
            self.module.body.block.add_op(load_data_func)
            shift_buffer_func = FuncOp.external(
                "shift_buffer",
                [
                    LLVMPointerType.typed(data_stream.elem_type),
                    LLVMPointerType.typed(stencil_stream.elem_type),
                    i32,
                    i32,
                    i32,
                ],
                [],
            )
            self.module.body.block.add_op(shift_buffer_func)

            self.set_load_data_declaration = True


@dataclass
class CallShiftBuffer(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: Rewriter, /):
        shift_buffer_call = Call.get("shift_buffer", [], [])


@dataclass
class HLSConvertStencilToLLMLIRPass(ModulePass):
    name = "hls-convert-stencil-to-ll-mlir"

    target: Literal["cpu", "gpu"] = "cpu"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets: dict[ReturnOp, list[SSAValue | None]] = return_target_analysis(
            op
        )

        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # FuncArgsToLLVMPtr(),
                    # ExtractForBody(op)
                    # CallShiftBuffer()
                    StencilExternalLoadToHLSExternalLoad(op)
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
