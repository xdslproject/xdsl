from dataclasses import dataclass
from typing import TypeVar
from warnings import warn

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import Block, BlockArgument, MLContext, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin, cf, gpu

from xdsl.dialects.experimental.stencil import AccessOp, ApplyOp, CastOp, FieldType, IndexAttr, LoadOp, ReturnOp, StoreOp, TempType
from xdsl.utils.exceptions import VerifyException

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
    input_type: FieldType[_TypeElement] | TempType[_TypeElement]
) -> MemRefType[_TypeElement]:
    dims = [i.value.data for i in input_type.shape.data]

    return MemRefType.from_element_type_and_shape(input_type.element_type,
                                                  dims)


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _TypeElement,
                                  lb: IndexAttr,
                                  ub: IndexAttr) -> MemRefType[_TypeElement]:
    # lb and ub defines the minimum and maximum coordinates of the resulting memref,
    # so its shape is simply ub - lb, computed here.
    dims = IndexAttr.size_from_bounds(lb, ub)

    return MemRefType.from_element_type_and_shape(memref_element_type, dims)


@dataclass
class CastOpToMemref(RewritePattern):

    return_target: dict[ReturnOp, CastOp | memref.Cast]
    gpu: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):

        assert isinstance(op.field.typ, FieldType | MemRefType)
        field_typ: FieldType[Attribute] | MemRefType[Attribute] = op.field.typ

        result_typ = GetMemRefFromFieldWithLBAndUB(field_typ.element_type,
                                                   op.lb, op.ub)

        cast = memref.Cast.get(op.field, result_typ)

        for k, v in self.return_target.items():
            if v == op:
                self.return_target[k] = cast

        if self.gpu:
            unranked = memref.Cast.get(
                cast.dest,
                memref.UnrankedMemrefType.from_type(field_typ.element_type))
            register = gpu.HostRegisterOp.from_memref(unranked.dest)
            rewriter.insert_op_after_matched_op([unranked, register])
        rewriter.replace_matched_op(cast)


class StoreOpCleanup(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):

        owner = op.temp.owner

        assert isinstance(owner, ApplyOp | LoadOp)

        owner.attributes['lb'] = IndexAttr.min(op.lb, owner.lb)
        owner.attributes['ub'] = IndexAttr.max(op.ub, owner.ub)

        rewriter.erase_matched_op()
        pass


@dataclass
class ReturnOpToMemref(RewritePattern):

    return_target: dict[ReturnOp, CastOp | memref.Cast]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp)

        cast = self.return_target[op]
        assert isinstance(cast, CastOp)

        offsets = cast.lb
        assert isinstance(offsets, IndexAttr)

        assert (block := op.parent_block()) is not None

        off_const_ops = [
            arith.Constant.from_int_and_width(-x.value.data,
                                              builtin.IndexType())
            for x in offsets.array.data
        ]

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(block.args, off_const_ops)
        ]

        load = memref.Store.get(op.arg, cast.result, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load])


class LoadOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        cast = op.field.owner
        assert isinstance(cast, CastOp)

        if IndexAttr.min(cast.lb, op.lb) != cast.lb:
            raise VerifyException(
                "The stencil computation requires a field with lower bound at least "
                f"{op.lb}, got {cast.lb}")

        rewriter.replace_matched_op([], list(cast.results))


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter):

    def access_shape_infer_walk(access: Operation) -> None:
        assert (op.lb is not None) and (op.ub is not None)
        if not isinstance(access, AccessOp):
            return
        assert isinstance(access.temp, BlockArgument)
        temp_owner = op.args[access.temp.index].owner

        assert isinstance(temp_owner, LoadOp | ApplyOp)

        temp_owner.attributes['lb'] = IndexAttr.min(op.lb + access.offset,
                                                    temp_owner.lb)
        temp_owner.attributes['ub'] = IndexAttr.max(op.ub + access.offset,
                                                    temp_owner.ub)

    op.walk(access_shape_infer_walk)

    assert (op.lb is not None) and (op.ub is not None)

    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.blocks[0]

    for arg in entry.args:
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.replace_operand(use.index, op.args[use.index])
        entry.erase_arg(arg)

    dim = len(op.lb.array.data)

    for _ in range(dim):
        rewriter.insert_block_argument(entry, 0, builtin.IndexType())

    return rewriter.move_region_contents_to_new_regions(op.region)


class ApplyOpToLaunch(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):

        assert (op.lb is not None) and (op.ub is not None)

        body = prepare_apply_body(op, rewriter)
        dim = len(op.lb.array.data)

        # This is naive and probably not robust.
        # Basically, on CUDA (modern hardware), blocks can host at most 1024 threads.
        # This is saying "If I have one dimension, ask for 1024x1x1 threads",
        # and 32x32x1=1024 for 2 dims, and 8x8x8=512 for 3 dims.
        threads_per_dim = [[1024, 32, 8][dim - 1]] * dim + [1] * (3 - dim)
        cst_tpd = [
            arith.Constant.from_int_and_width(tpd, builtin.IndexType())
            for tpd in threads_per_dim
        ]

        # These are the actual computation bounds, as int
        dims = IndexAttr.size_from_bounds(op.lb, op.ub)
        # Just prepare a 'one' index constant
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())

        # These are the computation bounds as "arith.constant"s
        bounds_cst = [
            arith.Constant.from_int_and_width(dims[i], builtin.IndexType())
            for i in range(dim)
        ]

        # We ceil-divide those bounds by the nunber of threads for this dimension
        # To get the number of blocks
        blocks_divs = [
            arith.CeilDivUI.get(bound, tpd)
            for bound, tpd in zip(bounds_cst, cst_tpd)
        ]

        # We build the actual gpu.launch
        launch = gpu.LaunchOp.get(body, blocks_divs + [one] * (3 - dim),
                                  cst_tpd)

        # We create a basic block, to compute thread indexes in
        index_compute_block = Block.from_arg_types([builtin.IndexType()] * 12)
        block_mul = [
            arith.Muli.get(index_compute_block.args[i],
                           index_compute_block.args[9 + i]) for i in range(dim)
        ]
        thread_add = [
            arith.Addi.get(index_compute_block.args[3 + i], block_mul[i])
            for i in range(dim)
        ]

        # We cast thread indices to i64 for comparison
        thread_int = [
            arith.IndexCastOp.get(ta, builtin.IntegerType.from_width(64))
            for ta in thread_add
        ]
        # Same for bounds
        bounds_int = [
            arith.IndexCastOp.get(bc, builtin.IntegerType.from_width(64))
            for bc in bounds_cst
        ]

        # We check that the indices are inbound
        cmpis: list[arith.Cmpi | arith.AndI] = [
            arith.Cmpi.from_mnemonic(ti, bc, "ult")
            for ti, bc in zip(thread_int, bounds_int)
        ]

        # We "arrith.andi" all those inbound checks
        if len(cmpis) >= 2:
            cmpis.append(arith.AndI.get(cmpis[0], cmpis[1]))
        for i in range(2, len(cmpis) - 1):
            cmpis.append(arith.AndI.get(cmpis[i], cmpis[-1]))

        # Then we simply conditionally jump to the kernel's body if we are inbound.
        else_block = Block.from_ops([gpu.TerminatorOp.get()])
        body_branch = cf.ConditionalBranch.get(cmpis[-1], body.blocks[0],
                                               list(thread_add), else_block,
                                               [])

        # We just put the right ops in the index computation blocks
        index_compute_block.add_ops([
            *block_mul, *thread_add, *thread_int, *bounds_int, *cmpis,
            body_branch
        ])

        # We insert the index computation block as the entry block of the gpu.launch body
        body.insert_block(index_compute_block, 0)
        body.add_block(else_block)
        # We add the right terminator to the gpu.launch kernel body
        launch.body.blocks[1].add_op(gpu.TerminatorOp.get())

        # We replace the stencil.apply with the gpu.launch!
        rewriter.insert_op_before_matched_op(
            [*cst_tpd, *bounds_cst, *blocks_divs, one, launch])
        rewriter.erase_matched_op(safe_erase=False)


class ApplyOpToParallel(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):

        assert (op.lb is not None) and (op.ub is not None)

        body = prepare_apply_body(op, rewriter)
        dim = len(op.lb.array.data)

        #Then create the corresponding scf.parallel
        dims = IndexAttr.size_from_bounds(op.lb, op.ub)
        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in dims
        ]

        # Move the body to the loop
        body.blocks[0].add_op(scf.Yield.get())
        p = scf.ParallelOp.get(lowerBounds=[zero] * dim,
                               upperBounds=upperBounds,
                               steps=[one] * dim,
                               body=body)

        # Replace with the loop and necessary constants.
        rewriter.insert_op_before_matched_op([zero, one, *upperBounds, p])
        rewriter.erase_matched_op(False)


class AccessOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):

        load = op.temp.owner
        assert isinstance(load, LoadOp)

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        assert isinstance(load.lb, IndexAttr)
        assert isinstance(load.field.owner, CastOp)
        memref_offset = (op.offset - load.field.owner.lb).array.data
        off_const_ops = [
            arith.Constant.from_int_and_width(x.value.data,
                                              builtin.IndexType())
            for x in memref_offset
        ]

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(block.args, off_const_ops)
        ]

        load = memref.Load.get(load.res, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load],
                                    [load.res])


class StencilTypeConversionFuncOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = []
        for arg in op.body.blocks[0].args:
            if isinstance(arg.typ, FieldType):
                typ: FieldType[Attribute] = arg.typ
                memreftyp = GetMemRefFromField(typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)
            else:
                inputs.append(arg.typ)

        op.attributes["function_type"] = FunctionType.from_lists(
            inputs, list(op.function_type.outputs.data))


def ConvertStencilToGPU(ctx: MLContext, module: ModuleOp):

    return_target: dict[ReturnOp, CastOp | memref.Cast] = {}

    def map_returns(op: Operation) -> None:
        if not isinstance(op, ReturnOp):
            return

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        res = list(apply.res)[0]

        if (len(res.uses) > 1) or (not isinstance(
            (store := list(res.uses)[0].operation), StoreOp)):
            warn("Only single store result atm")
            return

        cast = store.field.owner
        assert isinstance(cast, CastOp)

        return_target[op] = cast

    module.walk(map_returns)

    the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier([
        ApplyOpToLaunch(),
        StencilTypeConversionFuncOp(),
        CastOpToMemref(return_target, gpu=True),
        LoadOpToMemref(),
        AccessOpToMemref(),
        ReturnOpToMemref(return_target),
        StoreOpCleanup()
    ]),
                                        apply_recursively=False,
                                        walk_reverse=True)
    the_one_pass.rewrite_module(module)


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):

    return_target: dict[ReturnOp, CastOp | memref.Cast] = {}

    def map_returns(op: Operation) -> None:
        if not isinstance(op, ReturnOp):
            return

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        res = list(apply.res)[0]

        if (len(res.uses) > 1) or (not isinstance(
            (store := list(res.uses)[0].operation), StoreOp)):
            warn("Only single store result atm")
            return

        cast = store.field.owner
        assert isinstance(cast, CastOp)

        return_target[op] = cast

    module.walk(map_returns)

    the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier([
        ApplyOpToParallel(),
        StencilTypeConversionFuncOp(),
        CastOpToMemref(return_target),
        LoadOpToMemref(),
        AccessOpToMemref(),
        ReturnOpToMemref(return_target),
        StoreOpCleanup()
    ]),
                                        apply_recursively=False,
                                        walk_reverse=True)
    the_one_pass.rewrite_module(module)
