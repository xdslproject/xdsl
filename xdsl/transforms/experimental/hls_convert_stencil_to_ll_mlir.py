from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import MLContext, Operation, OpResult
from xdsl.dialects.builtin import i32, f64
from xdsl.dialects.func import FuncOp, Call
from xdsl.dialects import arith, builtin
from xdsl.dialects.arith import Constant

from xdsl.dialects.stencil import ExternalLoadOp

from xdsl.dialects.experimental.hls import (
    HLSStream,
)

from xdsl.dialects.memref import MemRefType

from xdsl.passes import ModulePass

from xdsl.utils.hints import isa

from xdsl.dialects.llvm import LLVMPointerType, LLVMStructType, LLVMArrayType

from xdsl.ir.core import BlockArgument


# TODO docstrings and comments
@dataclass
class StencilExternalLoadToHLSExternalLoad(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op
        self.set_load_data_declaration = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        assert isinstance(op.field, OpResult)
        field = op.field

        # Find the llvm.ptr to external memory that genrates the argument to the stencil.external_load. For PSyclone, this is
        # an argument to the parent function. TODO: this might need to be tested and generalised for other codes. Also, we are
        # considering that the function argument will be the second to insertvalue, but we're walking up trhough the second to
        # avoid bumping into arith.constants (see the mlir ssa).
        new_op = field
        func_arg = None

        while not isa(func_arg, BlockArgument):
            assert isinstance(new_op.owner, Operation)
            func_arg = new_op.owner.operands[-1]
            new_op = new_op.owner.operands[0]

        if isa(func_arg.typ, LLVMPointerType):
            func_arg_elem_type = func_arg.typ.type
        else:
            func_arg_elem_type = func_arg.typ

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

        assert isinstance(field.typ, MemRefType)
        shape = field.typ.get_shape()
        shape_x = Constant.from_int_and_width(shape[0], i32)
        shape_y = Constant.from_int_and_width(shape[1], i32)
        shape_z = Constant.from_int_and_width(shape[2], i32)

        two_int = Constant.from_int_and_width(2, i32)
        shift_shape_x = arith.Subi(shape_x, two_int)
        data_stream = HLSStream.get(struct_type)
        stencil_stream = HLSStream.get(struct_stencil_type)

        threedload_call = Call.get(
            "load_data", [func_arg, data_stream, shape_x, shape_y, shape_z], []
        )

        shift_buffer_call = Call.get(
            "shift_buffer",
            [data_stream, stencil_stream, shift_shape_x, shape_y, shape_z],
            [],
        )

        rewriter.insert_op_before_matched_op(
            [
                data_stream,
                stencil_stream,
                shape_x,
                shape_y,
                shape_z,
                two_int,
                shift_shape_x,
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
class HLSConvertStencilToLLMLIRPass(ModulePass):
    name = "hls-convert-stencil-to-ll-mlir"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op)]),
            apply_recursively=False,
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
