from typing import Iterable

from xdsl.ir import SSAValue, Attribute, MLContext, Operation
from xdsl.dialects import func, print, builtin, arith, llvm
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)

from xdsl.passes import ModulePass


i8 = builtin.IntegerType.from_width(8)


def _format_string_spec_from_print_op(op: print.PrintLnOp) -> Iterable[str | SSAValue]:
    """
    Translates the op:
    print.println "val = {}, val2 = {}", %1 : i32, %2 : f32

    into this sequence:
    ["val = ", %1, ", val2 = ", %2]

    Empty string parts are omitted.
    """
    format_str = op.format_str.data.split("{}")
    args = iter(op.format_vals)

    for part in format_str[:-1]:
        if part != "":
            yield part
        yield next(args)
    if format_str[-1] != "":
        yield format_str[-1]


def _format_str_for_typ(t: Attribute):
    match t:
        case builtin.f64:
            return "%f"
        case builtin.i32:
            return "%i"
        case builtin.i64:
            return "%li"
    raise ValueError(f"Cannot find printf code for {t}")


class PrintlnOpToPrintfCall(RewritePattern):
    printf_prefix: str
    printf_count: int

    def __init__(self):
        self.printf_prefix = "printf"
        self.printf_count = 0

    def _construct_global(self, val: str):
        self.printf_count += 1
        data = val.encode() + b"\x00"
        return llvm.GlobalOp.get(
            llvm.LLVMArrayType.from_size_and_type(len(data), i8),
            f"{self.printf_prefix}_data_{self.printf_count}",
            constant=True,
            linkage="internal",
            value=builtin.DenseArrayBase.create_dense_int_or_index(i8, data),
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: print.PrintLnOp, rewriter: PatternRewriter, /):
        format_str = ""
        args: list[SSAValue] = []
        casts: list[Operation] = []
        for part in _format_string_spec_from_print_op(op):
            if isinstance(part, str):
                format_str += part
            elif isinstance(part.typ, builtin.IndexType):
                # index must be cast to fixed bitwidth (I think) before printing
                casts.append(new_val := arith.IndexCastOp.get(part, builtin.i64))
                args.append(new_val.result)
                format_str += "%li"
            elif part.typ == builtin.f32:
                # f32 must be promoted to f64 before printing
                casts.append(new_val := arith.ExtFOp.get(part, builtin.f64))
                args.append(new_val.result)
                format_str += "%f"
            else:
                args.append(part)
                format_str += _format_str_for_typ(part.typ)

        # TODO: create the global thingy and load the address here
        rewriter.replace_matched_op(
            casts
            + [
                str_data := self._construct_global(format_str + "\n"),
                ptr := llvm.AddressOfOp.get(
                    str_data.sym_name, llvm.LLVMPointerType.typed(str_data.global_type)
                ),
                func.Call.get("printf", [ptr.result, *args], []),
            ]
        )


class PrintToPintf(ModulePass):
    name = "print-to-printf"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([PrintlnOpToPrintfCall()])
        ).rewrite_module(op)
