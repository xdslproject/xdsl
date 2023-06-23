from dataclasses import dataclass
from typing import Iterable, Sequence
import hashlib
import re

from utils.hints import isa
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
    val = re.sub(r"[^A-Za-z0-9_]+", "", val).strip("_")
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


class PrintToPrintf(ModulePass):
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


def _emit_putch_for_str(string: str) -> Iterable[Operation]:
    for c in string.encode():
        val = arith.Constant.from_int_and_width(c, 32)
        yield val
        yield llvm.CallOp("putchar", val)


def _emit_itoa_inline(number: SSAValue) -> Iterable[Operation]:
    """
    Emit the equivalent of:
    def itoa_inline(num: int):
        if num == 0:
            putchar("0")
            return

        if num < 0:
            putchar("-")
            itoa_inline(-num)
            return

        size = ceil(log10(num + 1))

        for i in range(size):
            pos = size - i - 1
            digit = (num // (10**pos)) % 10
            putchar(digit)


    In MLIR this is:

    module{
        func.func private @putchar(i32) -> i32

        func.func @main() -> (i32){
            %in = arith.constant 2323: i32
            %zero = arith.constant 0 : i32
            %is_negative = arith.cmpi slt, %in, %zero : i32
            %is_zero = arith.cmpi eq, %in, %zero : i32
            // If the value is zero, just print one zero
            // Otherwise log10 does not work
            %return_value = scf.if %is_zero -> (i32) {
                 %minus_one = arith.constant -1 : i32
                 %neg_in = arith.muli %minus_one, %in : i32
                 // Zero in ascii is 48 (decimal)
                 %ascii_zero = arith.constant 48 : i32
                 %return_value_zero = func.call @putchar(%ascii_zero) : (i32) -> (i32)
                 scf.yield %return_value_zero: i32
            } else {
                %in_positive = scf.if %is_negative -> (i32) {
                    // If the value is negative, print the minus
                    // minus in ascii is 45 (decimal)
                    %ascii_minus = arith.constant 45 : i32
                    %test = func.call @putchar(%ascii_minus) : (i32) -> (i32)
                    // continue with the positive number
                    %neg_in = arith.subi %zero, %in : i32
                    scf.yield %neg_in: i32
                } else {
                    scf.yield %in : i32
                }
                // If the value is positive, print each character with putchar
                // size = ceil(log10(num + 1))
                %one = arith.constant 1 : i32
                %plus_one = arith.addi %in_positive, %one : i32
                %plus_one_fp = arith.uitofp %plus_one : i32 to f32
                %size = math.log10 %plus_one_fp : f32
                %size_ceiled = math.ceil %size : f32
                %size_int = arith.fptoui %size_ceiled : f32 to i32
                scf.for %index_var=%zero to %size_int step %one : i32 {
                    // Print out the most significant digit first
                    //
                    // for i in range(size):
                    //   pos = size - i - 1
                    %size_minus_one = arith.subi %size_int, %one : i32
                    %position = arith.subi %size_minus_one, %index_var: i32

                    // digit = (num // (10**pos)) % 10
                    %ten = arith.constant 10 : i32
                    %i_0 = "math.ipowi"(%ten, %position): (i32, i32)-> i32
                    %i_1 = arith.divui %in_positive, %i_0 : i32
                    %digit = arith.remui %i_1, %ten : i32
                    %ascii_offset = arith.constant 48 : i32
                    %char = arith.addi %digit, %ascii_offset : i32
                    %return_value = func.call @putchar(%char) : (i32) -> (i32)
                }
                // We can not return multiple values, so just return one instead
                scf.yield %one: i32
            }
        return %return_value: i32
        }
    }
    """
    return []


class PrintToPutchar(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: print.PrintLnOp, rewriter: PatternRewriter, /):
        parts = _format_string_spec_from_print_op(op)

        new_ops: list[Operation] = []

        for part in parts:
            if isinstance(part, str):
                new_ops.extend(_emit_putch_for_str(part))
                continue
            assert isinstance(part, SSAValue)

            if isinstance(part.owner, arith.Constant):
                if isa(part.owner.value, builtin.IntegerAttr | builtin.FloatAttr):
                    new_ops.extend(
                        _emit_putch_for_str(str(part.owner.value.value.data))
                    )
                    continue

            if isinstance(part.typ, builtin.IntegerType):
                new_ops.extend(_emit_itoa_inline(part))

            raise ValueError("Cannot putchar {}".format(part))

        # add the newline
        new_ops.extend(_emit_putch_for_str("\n"))

        rewriter.replace_matched_op(new_ops)


class PrintToPucharPass(ModulePass):
    name = "print-to-putchar"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(PrintToPutchar()).rewrite_module(op)

        # TODO: is this a "nice" thing to do or big no-no?
        op.body.block.add_ops(
            [
                llvm.FuncOp(
                    "putchar",
                    llvm.LLVMFunctionType([builtin.i32]),
                    linkage=llvm.LinkageAttr("external"),
                ),
            ]
        )
