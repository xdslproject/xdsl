from dataclasses import dataclass

from xdsl.builder import Builder, InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import affine, builtin, func, memref
from xdsl.dialects.experimental.hida_prim import MemoryKind, MemoryKindAttr
from xdsl.dialects.experimental.utils import (
    get_invariant_output_band,
    get_loop_bands_any_nchildren,
    hoist_constants,
    tile_loop,
)
from xdsl.dialects.func import FuncOp
from xdsl.ir import BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class PromoteInnerLoopToNode(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, node: FuncOp, rewriter: PatternRewriter):
        hoist_constants(node, rewriter)
        bands = get_loop_bands_any_nchildren(node.body.block)

        # print("N. BANDS: ", len(bands))

        # minimum_independent_band = get_minimum_independent_band(bands)
        # print("MIN IDPNT BAND: \n",minimum_independent_band[0], "\n")
        invariant_output_band = get_invariant_output_band(bands)
        # print("INVARIANT OUTPUT BAND: ", invariant_output_band[-1])

        tile_loop(invariant_output_band[0], 0, 2, rewriter, node)


@dataclass
class HoistLoadsIntoCopyNodes(RewritePattern):
    #########################################################################################
    # Generates a copy node for each load operation. The memref buffer is copied to on-chip
    # memory.  The idea is that these nodes can  be pipelined in a load-compute-store manner
    # whilst reducing the latency of the  computation.
    #########################################################################################

    module: builtin.ModuleOp
    copy_node_idx = 0

    def hoist_load_into_copy_node(
        self, load: affine.Load, rewriter: PatternRewriter, top: func.FuncOp
    ):
        parent_function = load.parent_op()
        while not isinstance(parent_function, FuncOp):
            parent_function = parent_function.parent_op()

        buf_type = load.memref.type
        onchip_buf_type = builtin.MemRefType(
            buf_type.element_type,
            buf_type.shape,
            memory_space=MemoryKindAttr(MemoryKind.BRAM_2P),
        )

        @Builder.region([buf_type, onchip_buf_type])
        def copy_body(builder: Builder, args: [BlockArgument, ...]):
            copy = memref.CopyOp(args[0], args[1])
            builder.insert(copy)
            builder.insert(func.Return())

        copy_node = FuncOp(
            f"copy_node_{__class__.copy_node_idx}",
            builtin.FunctionType.from_lists([buf_type, onchip_buf_type], []),
            copy_body,
        )
        __class__.copy_node_idx += 1

        copy_node.attributes["top_func"] = builtin.UnitAttr(
            []
        )  # NOTE: Needed by ScaleHLS to trigger the QoR estimation
        rewriter.insert_op(copy_node, InsertPoint.at_start(self.module.body.block))

        return copy_node

    def update_node_mem_space(self, node: FuncOp, mem_kind: MemoryKind):
        new_input_types: list[builtin.Attribute] = []
        for i in range(len(node.args)):
            assert isinstance(node.args[i].type, builtin.MemRefType)
            new_type = builtin.MemRefType(
                node.args[i].type.element_type,
                node.args[i].type.shape,
                memory_space=MemoryKindAttr(mem_kind),
            )
            node.args[i].type = new_type

            new_input_types.append(new_type)

        node.function_type = builtin.FunctionType.from_lists(new_input_types, [])

    @op_type_rewrite_pattern
    def match_and_rewrite(self, node: func.FuncOp, rewriter: PatternRewriter):
        memref_types = list(map(lambda t: t.type, node.args))

        @Builder.region(memref_types)
        def top_body(builder: Builder, args: [BlockArgument, ...]):
            builder.insert(func.Return())

        top = FuncOp("top", builtin.FunctionType.from_lists(memref_types, []), top_body)
        top.attributes["top_func"] = builtin.UnitAttr(
            []
        )  # NOTE: Needed by ScaleHLS to trigger the QoR estimation
        rewriter.insert_op(top, InsertPoint.at_start(self.module.body.block))

        node_arg_buf_map = dict()

        copy_nodes: list[func.FuncOp] = []
        call_copy_nodes: list[func.Call] = []
        all_loads = filter(lambda op: isinstance(op, affine.Load), node.walk())
        for load_idx, load in enumerate(all_loads):
            if load.memref in node.args:
                copy_node = self.hoist_load_into_copy_node(load, rewriter, top)
                copy_nodes.append(copy_node)

            assert isinstance(load.memref.type, builtin.MemRefType)
            buf_type = load.memref.type.element_type
            buf_shape = load.memref.type.shape
            buf = memref.Alloc.get(
                buf_type,
                shape=buf_shape,
                memory_space=MemoryKindAttr(MemoryKind.BRAM_2P),
            )
            rewriter.insert_op(buf, InsertPoint.at_start(top.body.block))

            call_copy_node = func.Call(
                copy_node.sym_name.data, [top.args[load_idx], buf.memref], []
            )
            call_copy_nodes.append(call_copy_node)
            rewriter.insert_op(call_copy_node, InsertPoint.after(buf))

            node_arg_buf_map[node.args[load_idx]] = buf

        buf_args = []
        for arg in node.args:
            buf_args.append(node_arg_buf_map[arg])

        node_call = func.Call(node.sym_name.data, buf_args, [])

        rewriter.insert_op(node_call, InsertPoint.before(top.body.block.last_op))

        self.update_node_mem_space(node, MemoryKind.BRAM_2P)


@dataclass(frozen=True)
class SplitDataflowNodes(ModulePass):
    name = "split-dataflow-nodes"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        split_dataflow_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([HoistLoadsIntoCopyNodes(op)]),
            GreedyRewritePatternApplier([PromoteInnerLoopToNode()]),
            apply_recursively=False,
            walk_reverse=False,
        )
        split_dataflow_pass.rewrite_module(op)
