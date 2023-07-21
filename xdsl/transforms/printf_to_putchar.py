from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import IndexType, ModuleOp, i32
from xdsl.dialects.experimental import math
from xdsl.dialects.printf import PrintCharOp, PrintIntOp
from xdsl.ir import Block, MLContext, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.printer import Printer


class LowerPrintCharToPutchar(RewritePattern):
    """
    Rewrite Pattern that rewrites printf.print_char to an
    external function call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintCharOp, rewriter: PatternRewriter, /):
        func_call = func.Call.get("putchar", [op.char], [i32])
        # Add empty new_results, since a result is necessary for linking
        # putchar, but the result does not exist anywhere.
        rewriter.replace_matched_op(func_call, new_results=[])


class ConvertPrintIntToItoa(RewritePattern):
    """
    Rewrite Pattern that rewrites printf.print_int to an
    inline_itoa function
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintIntOp, rewriter: PatternRewriter, /):
        func_call = func.Call.get("inline_itoa", [op.int], [])
        rewriter.replace_matched_op(func_call)


def get_inline_itoa():
    inline_itoa_func = func.FuncOp("inline_itoa", ((i32,), ()))
    with ImplicitBuilder(inline_itoa_func.body) as (integer,):
        zero = arith.Constant.from_int_and_width(0, i32)
        is_zero_block = Block()
        is_not_zero_block = Block()
        is_zero = arith.Cmpi.get(integer, zero, "eq")

        scf.If.get(
            is_zero,
            [],
            [is_zero_block],
            [is_not_zero_block],
        )
        # If the value is zero, just print one zero
        with ImplicitBuilder(is_zero_block):
            ascii_zero = arith.Constant.from_int_and_width(48, i32)
            # The ascii value for zero is 48 in decimal
            PrintCharOp((ascii_zero,))
            scf.Yield.get()
        # Otherwise continue with itoa
        with ImplicitBuilder(is_not_zero_block):
            is_negative_block = Block()
            is_positive_block = Block()
            is_negative = arith.Cmpi.get(integer, zero, "slt")

            # Either way get the absolute value of the number
            absolute_value = scf.If.get(
                is_negative, [i32], [is_negative_block], [is_positive_block]
            )
            # If the value is negative, print the minus sign, return abs value
            with ImplicitBuilder(is_negative_block):
                ascii_minus = arith.Constant.from_int_and_width(45, i32)
                PrintCharOp((ascii_minus,))
                negative_input = arith.Subi(zero, integer)
                scf.Yield.get(negative_input)
            # If the value is positive, just return value itself as abs value
            with ImplicitBuilder(is_positive_block):
                scf.Yield.get(integer)

            # Now print the digits of the absolute value

            # Get the amount of digits
            before_region = Region(Block(arg_types=(i32,)))
            after_region = Region(Block(arg_types=(i32,)))
            size_int = scf.While.get(
                (absolute_value,), (i32,), before_region, after_region
            )
            with ImplicitBuilder(before_region.blocks[0]) as (running_integer,):
                is_bigger_than_zero = arith.Cmpi.get(running_integer, zero, "slt")
                scf.Condition.get(is_bigger_than_zero, running_integer)
            with ImplicitBuilder(after_region.blocks[0]) as (running_integer,):
                ten = arith.Constant.from_int_and_width(10, 32)
                new_integer = arith.DivUI(running_integer, ten)
                scf.Yield.get(new_integer)
            zero_index = arith.Constant.from_int_and_width(0, IndexType())
            one_index = arith.Constant.from_int_and_width(1, IndexType())
            size_int_index = arith.IndexCastOp((size_int,), (IndexType(),))
            loop_body = Block(arg_types=(IndexType(),))

            # Print all from most significant to least
            scf.For.get(zero_index, size_int_index, one_index, (), loop_body)
            with ImplicitBuilder(loop_body) as (index_var,):
                one = arith.Constant.from_int_and_width(1, i32)
                size_minus_one = arith.Subi(size_int, one)
                index_var_int = arith.IndexCastOp((index_var,), (i32,))
                position = arith.Subi(size_minus_one, index_var_int)
                # digit = (num // (10**pos)) % 10
                ten = arith.Constant.from_int_and_width(10, i32)
                i_0 = math.IPowIOp((ten, position), (i32,))
                i_1 = arith.DivUI(absolute_value, i_0)
                digit = arith.RemUI(i_1, ten)
                # ascii value for zero is 48 in decimal
                ascii_offset = arith.Constant.from_int_and_width(48, i32)
                char = arith.Addi(digit, ascii_offset)
                PrintCharOp((char,))
                scf.Yield.get()
            scf.Yield.get()

        # Return from itoa function
        func.Return()

    return inline_itoa_func


if __name__ == "__main__":
    printer = Printer()
    printer.print(get_inline_itoa())


class LowerPrintCharToPutcharPass(ModulePass):
    name = "lower-printchar-to-putchar"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        print_int_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertPrintIntToItoa(),
                ]
            ),
            apply_recursively=True,
        )
        print_int_walker.rewrite_module(op)
        # This has to happen first, since inline itoa
        # contains some references to printf.print_char
        op.body.block.add_ops([get_inline_itoa()])
        # Add function implementation for inline_itoa
        # TODO Don't insert this if the operation is not there!
        print_char_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerPrintCharToPutchar()]),
            apply_recursively=True,
        )
        print_char_walker.rewrite_module(op)
        # Add external putchar reference
        # TODO Don't insert this if the operation is not there!
        op.body.block.add_ops([func.FuncOp.external("putchar", [i32], [i32])])
