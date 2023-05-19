from collections import Counter
from dataclasses import dataclass, field

from xdsl.ir import MLContext, Operation, Block, Region
from xdsl.dialects.builtin import (
    AnyFloatAttr,
    AnyIntegerAttr,
    DenseIntOrFPElementsAttr,
    IntegerType,
    ModuleOp,
    VectorType,
)
from xdsl.dialects import llvm
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.dialects import riscv, riscv_func
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.hints import isa


class AddSections(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
        # bss stands for block starting symbol
        heap_section = riscv.DirectiveOp(
            ".bss",
            None,
            Region(
                Block(
                    [
                        riscv.LabelOp("heap"),
                        riscv.DirectiveOp(".space", f"{1024}"),  # 1kb
                    ]
                )
            ),
        )
        data_section = riscv.DirectiveOp(".data", None, Region(Block()))
        text_section = riscv.DirectiveOp(
            ".text", None, rewriter.move_region_contents_to_new_regions(op.regions[0])
        )

        op.body.add_block(Block([heap_section, data_section, text_section]))


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
        assert op.value is None, "Only support return with no arguments for now"

        rewriter.replace_matched_op(riscv_func.ReturnOp())


class LowerFAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FAddOp, rewriter: PatternRewriter):
        assert isinstance(op.lhs.typ, VectorType), "Only support vector add for now"
        rewriter.replace_matched_op(
            [
                res := riscv.CustomEmulatorInstructionOp(
                    "vector.copy", (op.lhs,), (riscv.RegisterType(riscv.Register()),)
                ),
                riscv.CustomEmulatorInstructionOp(
                    "vector.add", (res.results[0], op.rhs), ()
                ),
            ],
            [res.results[0]],
        )


class LowerFMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FMulOp, rewriter: PatternRewriter):
        assert isinstance(op.lhs.typ, VectorType), "Only support vector mul for now"
        rewriter.replace_matched_op(
            [
                res := riscv.CustomEmulatorInstructionOp(
                    "vector.copy", (op.lhs,), (riscv.RegisterType(riscv.Register()),)
                ),
                riscv.CustomEmulatorInstructionOp(
                    "vector.mul", (res.results[0], op.rhs), ()
                ),
            ],
            [res.results[0]],
        )


@dataclass
class DataDirectiveRewritePattern(RewritePattern):
    _data_directive: riscv.DirectiveOp | None = None
    _counter: Counter[str] = field(default_factory=Counter)

    def data_directive(self, op: Operation) -> riscv.DirectiveOp:
        """
        Relies on the data directive being inserted earlier
        """
        if self._data_directive is None:
            module_op = op.get_toplevel_object()
            assert isinstance(
                module_op, ModuleOp
            ), f"The top level object of {str(op)} must be a ModuleOp"

            for op in module_op.body.blocks[0].ops:
                if not isinstance(op, riscv.DirectiveOp):
                    continue
                if op.directive.data != ".data":
                    continue
                self._data_directive = op

            assert self._data_directive is not None

        return self._data_directive

    def label(self, func_name: str) -> str:
        key = func_name
        count = self._counter[key]
        self._counter[key] += 1
        return f"{key}.{count}"

    def func_name_of_op(self, op: Operation) -> str:
        region = op.parent_region()
        assert region is not None
        func_op = region.parent_op()
        assert isinstance(func_op, riscv_func.FuncOp)
        return func_op.func_name.data

    def add_data(self, op: Operation, label: str, data: list[int]):
        encoded_data = ", ".join(hex(el) for el in data)
        self.data_directive(op).regions[0].blocks[0].add_ops(
            [riscv.LabelOp(label), riscv.DirectiveOp(".word", encoded_data)]
        )


class LowerConstantOp(DataDirectiveRewritePattern):
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
            riscv.CustomEmulatorInstructionOp(
                op.intrin, op.operands, [res.typ for res in op.results]
            )
        )


class LowerUndefOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.UndefOp, rewriter: PatternRewriter):
        # TODO: clean up stack when returning from function

        res_type = op.res.typ
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
        res_type = op.res.typ
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
                op.value,
                op.container,
                offset,
                comment=f"Set {type_name} @ {position}",
            ),
            new_results=[op.container],
        )


class LowerExtractValueOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.InsertValueOp, rewriter: PatternRewriter):
        res_type = op.res.typ
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
        PatternRewriteWalker(AddSections()).rewrite_module(op)
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
