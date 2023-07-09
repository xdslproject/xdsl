from xdsl.dialects import llvm, riscv, riscv_func
from xdsl.dialects.builtin import (
    AnyFloatAttr,
    AnyIntegerAttr,
    DenseIntOrFPElementsAttr,
    IntegerType,
    ModuleOp,
    VectorType,
)
from xdsl.ir import MLContext
from xdsl.ir.core import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.hints import isa

from .setup_riscv_pass import DataDirectiveRewritePattern


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        # TODO: add support for user defined functions
        assert name == "main", "Only support lowering main function for now"

        region = op.regions[0]

        # insert a heap pointer at the start of every function
        # TODO: replace with insert_op_at_start
        first_op = region.blocks[0].first_op
        assert first_op is not None
        heap = riscv.LiOp("heap")
        rewriter.insert_op_before(heap, first_op)
        rewriter.insert_op_after(
            riscv.AddiOp(
                heap,
                1020,
                rd=riscv.Registers.SP,
                comment="stack grows from the top of the heap",
            ),
            heap,
        )

        # create riscv func op with same ops
        riscv_op = riscv_func.FuncOp(
            name, rewriter.move_region_contents_to_new_regions(region)
        )

        rewriter.replace_matched_op(riscv_op)


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ReturnOp, rewriter: PatternRewriter):
        # TODO: add support for optional argument
        assert op.arg is None, "Only support return with no arguments for now"

        rewriter.replace_matched_op(riscv_func.ReturnOp(()))


class LowerFAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FAddOp, rewriter: PatternRewriter):
        assert isinstance(op.lhs.type, VectorType), "Only support vector add for now"
        rewriter.replace_matched_op(
            [
                res := riscv.CustomAssemblyInstructionOp(
                    "vector.copy", (op.lhs,), (riscv.RegisterType(riscv.Register()),)
                ),
                riscv.CustomAssemblyInstructionOp(
                    "vector.add", (res.results[0], op.rhs), ()
                ),
            ],
            [res.results[0]],
        )


class LowerFMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FMulOp, rewriter: PatternRewriter):
        assert isinstance(op.lhs.type, VectorType), "Only support vector mul for now"
        rewriter.replace_matched_op(
            [
                res := riscv.CustomAssemblyInstructionOp(
                    "vector.copy", (op.lhs,), (riscv.RegisterType(riscv.Register()),)
                ),
                riscv.CustomAssemblyInstructionOp(
                    "vector.mul", (res.results[0], op.rhs), ()
                ),
            ],
            [res.results[0]],
        )


class LowerConstantOp(DataDirectiveRewritePattern):
    def func_name_of_op(self, op: Operation) -> str:
        region = op.parent_region()
        assert region is not None
        func_op = region.parent_op()
        assert isinstance(func_op, riscv_func.FuncOp)
        return func_op.sym_name.data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ConstantOp, rewriter: PatternRewriter):
        """
        Vectors are represented in memory as an n+1 array of int32, where the first
        entry is the count of the vector
        """
        if not isinstance(op.value, DenseIntOrFPElementsAttr):
            raise NotImplementedError("Only support vector constants for now")

        def get_value(el: AnyIntegerAttr | AnyFloatAttr) -> int:
            data = el.value.data
            if isinstance(data, float):
                if float(int(data)) != data:
                    raise ValueError(
                        f"Cannot store constant with non-integer value {data}"
                    )
            return int(data)

        data = [get_value(el) for el in op.value.data]
        label = self.label(self.func_name_of_op(op))

        self.add_data(op, label, [len(data), *data])
        rewriter.replace_matched_op(riscv.LiOp(label))


class LowerCallIntrinsicOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.CallIntrinsicOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            riscv.CustomAssemblyInstructionOp(
                op.intrin, op.operands, [res.type for res in op.results]
            )
        )


class LowerUndefOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.UndefOp, rewriter: PatternRewriter):
        # TODO: clean up stack when returning from function

        res_type = op.res.type
        assert isinstance(res_type, llvm.LLVMStructType)
        type_name = res_type.struct_name
        type_size = 0
        for element_type in res_type.types:
            assert isinstance(
                element_type, IntegerType
            ), "Only support integer type elements for now"
            width = element_type.width.data
            assert not width % 8
            type_size += width // 8

        rewriter.replace_matched_op(
            [
                # Get stack pointer stored in `SP`
                sp := riscv.GetRegisterOp(riscv.Registers.SP),
                riscv.CommentOp(
                    f"Reserve {type_size} bytes on stack for element type {type_name}"
                ),
                riscv.AddiOp(sp, -type_size, rd=riscv.Registers.SP),
            ]
        )


class LowerInsertValueOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.InsertValueOp, rewriter: PatternRewriter):
        res_type = op.res.type
        assert isinstance(res_type, llvm.LLVMStructType)
        type_name = res_type.struct_name

        assert len(op.position.data) == 1, "Only support shallow position for now"

        position = op.position.data.data[0].data

        offset = 0
        for element_type in res_type.types.data[:position]:
            assert isinstance(
                element_type, IntegerType
            ), "Only support integer type elements for now"
            width = element_type.width.data
            assert not width % 8
            offset += width // 8

        # llvm.InsertValueOp returns a result, which is a new SSA value representing the
        # updated container. riscv.SwOp doesn't return a result, so we can't just replace
        # one by the other. It might technically be better to create a new register value,
        # stored in the same register as the input, to help keep track of the value,
        # but we just forward the input for now.

        rewriter.replace_matched_op(
            riscv.SwOp(
                op.container,
                op.value,
                offset,
                comment=f"Set {type_name} @ {position}",
            ),
            new_results=[op.container],
        )


class LowerExtractValueOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.InsertValueOp, rewriter: PatternRewriter):
        res_type = op.res.type
        assert isinstance(res_type, llvm.LLVMStructType)
        type_name = res_type.struct_name

        assert len(op.position.data) == 1, "Only support shallow position for now"

        position = op.position.data.data[0].data

        offset = 0
        for element_type in res_type.types.data[:position]:
            assert isinstance(
                element_type, IntegerType
            ), "Only support integer type elements for now"
            width = element_type.width.data
            assert not width % 8
            offset += width // 8

        rewriter.replace_matched_op(
            riscv.LwOp(
                op.container,
                offset,
                comment=f"Get {type_name} @ {position}",
            )
        )


class LowerAllocaOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AllocaOp, rewriter: PatternRewriter):
        assert op.alignment == 32
        assert isa(op.size, AnyIntegerAttr)
        size = op.size.value.data

        rewriter.replace_matched_op(
            [
                # Get stack pointer stored in `SP`
                sp := riscv.GetRegisterOp(riscv.Registers.SP),
                riscv.CommentOp(f"Allocate {size} bytes on stack"),
                riscv.AddiOp(sp, -size, rd=riscv.Registers.SP),
                # Copy stack address to new register
                riscv.MVOp(sp),
            ]
        )


class LowerLLVM(ModulePass):
    name = "llvm-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerFuncOp()).rewrite_module(op)
        PatternRewriteWalker(LowerReturnOp()).rewrite_module(op)
        PatternRewriteWalker(LowerFAddOp()).rewrite_module(op)
        PatternRewriteWalker(LowerFMulOp()).rewrite_module(op)
        PatternRewriteWalker(LowerConstantOp()).rewrite_module(op)
        PatternRewriteWalker(LowerCallIntrinsicOp()).rewrite_module(op)
        PatternRewriteWalker(LowerUndefOp()).rewrite_module(op)
        PatternRewriteWalker(LowerInsertValueOp()).rewrite_module(op)
        PatternRewriteWalker(LowerExtractValueOp()).rewrite_module(op)
        PatternRewriteWalker(LowerAllocaOp()).rewrite_module(op)

        dce(op)
