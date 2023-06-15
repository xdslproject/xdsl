from dataclasses import dataclass
from typing import Iterable, Sequence
import hashlib
import re

from xdsl.ir import SSAValue, Attribute, MLContext, Operation
from xdsl.dialects import print, builtin, arith, llvm
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
)

from xdsl.passes import ModulePass


i8 = builtin.IntegerType(8)


def legalize_str(val: str):
    """
    Takes any string and legalizes it to be a global llvm symbol.
    (for the strictes possible interpreation of this)

     - Replaces all whitespaces and dots with _
     - Deletes all non ascii alphanumerical characters

    The resulting string consists only of ascii letters, underscores and digits
    """
    val = re.sub(r"(\s+|\.)", "_", val)
    val = re.sub(r"[^A-Za-z0-9_]+", "", val).rstrip("_")
    return val


def _key_from_str(val: str) -> str:
    """
    Generate a symbol name form any given string.

    Takes the first ten letters of the string plus it's sha1 hash to create a
    (pretty much) globally unique symbol name.
    """
    h = hashlib.new("sha1")
    h.update(val.encode())
    return f"{legalize_str(val[:10])}_{h.hexdigest()}"


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
        case _:
            raise ValueError(f"Cannot find printf code for {t}")


class PrintlnOpToPrintfCall(RewritePattern):
    collected_global_symbs: dict[str, llvm.GlobalOp]

    def __init__(self):
        self.collected_global_symbs = dict()

    def _construct_global(self, val: str):
        """
        Constructs an llvm.global operation containing the string.
        """
        data = val.encode() + b"\x00"

        t_type = builtin.TensorType.from_type_and_list(i8, [len(data)])

        return llvm.GlobalOp.get(
            llvm.LLVMArrayType.from_size_and_type(len(data), i8),
            _key_from_str(val),
            constant=True,
            linkage="internal",
            value=builtin.DenseIntOrFPElementsAttr.from_list(t_type, data),
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: print.PrintLnOp, rewriter: PatternRewriter, /):
        format_str = ""
        args: list[SSAValue] = []
        casts: list[Operation] = []
        # make sure all arguments are in the format libc expects them to be
        # e.g. floats must be promoted to double before calling
        for part in _format_string_spec_from_print_op(op):
            if isinstance(part, str):
                format_str += part
            elif isinstance(part.typ, builtin.IndexType):
                # index must be cast to fixed bitwidth before printing
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

        globl = self._construct_global(format_str + "\n")
        self.collected_global_symbs[globl.sym_name.data] = globl

        rewriter.replace_matched_op(
            casts
            + [
                ptr := llvm.AddressOfOp.get(
                    globl.sym_name, llvm.LLVMPointerType.opaque()
                ),
                llvm.CallOp("printf", ptr.result, *args),
            ]
        )


@dataclass(frozen=True)
class AddExternalFunctionDecl(RewritePattern):
    name: str
    signature: llvm.LLVMFunctionType

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter, /):
        rewriter.insert_op_at_end(llvm.FuncOp(self.name, self.signature), op.body.block)


@dataclass(frozen=True)
class AddGlobalSymbols(RewritePattern):
    symbols: Sequence[llvm.GlobalOp]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter, /):
        rewriter.insert_op_at_end(self.symbols, op.body.block)


class PrintToPintf(ModulePass):
    name = "print-to-printf"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        add_printf_call = PrintlnOpToPrintfCall()

        PatternRewriteWalker(add_printf_call).rewrite_module(op)

        # TODO: is this a "nice" thing to do or big no-no?
        op.body.block.add_ops(
            [
                llvm.FuncOp(
                    "printf",
                    llvm.LLVMFunctionType(
                        [llvm.LLVMPointerType.opaque()], is_variadic=True
                    ),
                    linkage=llvm.LinkageAttr("external"),
                ),
                *add_printf_call.collected_global_symbs.values(),
            ]
        )
