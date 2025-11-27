import hashlib
import re
from collections.abc import Iterable

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, printf
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

i8 = builtin.IntegerType(8)


def legalize_str_for_symbol_name(val: str):
    """
    Takes any string and legalizes it to be a global llvm symbol.
    (for the strictest possible interpretation of this)

     - Replaces all whitespaces and dots with _
     - Deletes all non ascii alphanumerical characters
     - Strips all underscores from the start and end of the string

    The resulting string consists only of ascii letters, underscores and digits.

    This is a surjective mapping, meaning that multiple inputs will produce the same
    output. This function alone cannot be used to get a uniquely identifying global
    symbol name for a string!
    """
    val = re.sub(r"(\s+|\.)", "_", val)
    val = re.sub(r"[^A-Za-z0-9_]+", "", val).strip("_")
    return val


def _key_from_str(val: str) -> str:
    """
    Generate a symbol name from any given string.

    Takes the first ten letters of the string plus it's sha1 hash to create a
    (pretty much) globally unique symbol name.
    """
    h = hashlib.new("sha1")
    h.update(val.encode())
    return f"{legalize_str_for_symbol_name(val[:10])}_{h.hexdigest()}"


def _format_string_spec_from_print_op(
    op: printf.PrintFormatOp,
) -> Iterable[str | SSAValue]:
    """
    Translates the op:
    printf.print_format "val = {}, val2 = {}\n", %1 : i32, %2 : f32

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


def _format_str_for_type(t: Attribute):
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
        Constructs an llvm.global operation containing the string. Assigns a unique
        symbol name to the value that is derived from the string value.
        """
        data = val.encode() + b"\x00"

        t_type = builtin.TensorType(i8, [len(data)])

        return llvm.GlobalOp(
            llvm.LLVMArrayType.from_size_and_type(len(data), i8),
            _key_from_str(val),
            constant=True,
            linkage="internal",
            value=builtin.DenseIntOrFPElementsAttr(t_type, builtin.BytesAttr(data)),
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter, /):
        format_str = ""
        args: list[SSAValue] = []
        casts: list[Operation] = []
        # make sure all arguments are in the format libc expects them to be
        # e.g. floats must be promoted to double before calling
        for part in _format_string_spec_from_print_op(op):
            if isinstance(part, str):
                format_str += part
            elif isinstance(part.type, builtin.IndexType):
                # index must be cast to fixed bitwidth before printing
                casts.append(new_val := arith.IndexCastOp(part, builtin.i64))
                args.append(new_val.result)
                format_str += "%li"
            elif part.type == builtin.f32:
                # f32 must be promoted to f64 before printing
                casts.append(new_val := arith.ExtFOp(part, builtin.f64))
                args.append(new_val.result)
                format_str += "%f"
            else:
                args.append(part)
                format_str += _format_str_for_type(part.type)

        globl = self._construct_global(format_str)
        self.collected_global_symbs[globl.sym_name.data] = globl

        rewriter.replace_op(
            op,
            casts
            + [
                ptr := llvm.AddressOfOp(globl.sym_name, llvm.LLVMPointerType()),
                llvm.CallOp("printf", ptr.result, *args, variadic_args=len(args)),
            ],
        )


class PrintfToLLVM(ModulePass):
    name = "printf-to-llvm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        add_printf_call = PrintlnOpToPrintfCall()

        PatternRewriteWalker(add_printf_call).rewrite_module(op)

        if not add_printf_call.collected_global_symbs:
            return

        op.body.block.add_ops(
            [
                llvm.FuncOp(
                    "printf",
                    llvm.LLVMFunctionType([llvm.LLVMPointerType()], is_variadic=True),
                    linkage=llvm.LinkageAttr("external"),
                ),
                *add_printf_call.collected_global_symbs.values(),
            ]
        )
