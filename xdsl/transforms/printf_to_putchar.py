from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, func, math, scf
from xdsl.dialects.builtin import IndexType, IntegerType, ModuleOp, i32
from xdsl.dialects.printf import PrintCharOp, PrintIntOp
from xdsl.ir import Block, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

i8 = IntegerType(8)


class LowerPrintCharToPutchar(RewritePattern):
    """
    Rewrite Pattern that rewrites printf.print_char to an
    external function call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintCharOp, rewriter: PatternRewriter, /):
        func_call = func.CallOp("putchar", [op.char], [i32])
        cast_op = arith.ExtUIOp(op.char, i32)
        func_call = func.CallOp("putchar", [cast_op], [i32])
        # Add empty new_results, since a result is necessary for linking
        # putchar, but the result does not exist anywhere.
        rewriter.replace_op(op, [cast_op, func_call], new_results=[])


class ConvertPrintIntToItoa(RewritePattern):
    """
    Rewrite Pattern that rewrites printf.print_int to an
    mlir_itoa function
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintIntOp, rewriter: PatternRewriter, /):
        func_call = func.CallOp("mlir_itoa", [op.int], [])
        rewriter.replace_op(op, func_call)


def get_mlir_itoa():
    """
    This function returns an MLIR func.func that holds a routine that calls
    an external putchar implementation several times on all individual digits
    that make up the integer.

    Roughly it exists of the following steps:
        1. If the number is zero, print zero and exit
        2. Check if the number is negative: if so, print the minus sign
        3. Get the number of digits for the absolute value
        4. Go through the digits in reverse order and print them one by one

    Several things to note about this specfic implementation:
        * It does not require floating point arithmetic
        * It does not require a printf implementation, only putchar.
    """

    # get_number_of_digits implementation
    def get_number_of_digits(absolute_value: SSAValue) -> SSAValue:
        """
        Return the result of an MLIR scf.while op that gets the number of
        digits of a positive integer. Roughly goes like this in
        python pseudocode (e.g. integer is 420, routine should return 3):
        ```
        integer = 420
        digits = 0
        while(integer != 0):
            integer //= 10
            digits += 1
        return digits

        ```
        """
        before_block = Block(arg_types=(i32, i32))
        after_block = Block(arg_types=(i32, i32))
        digit_init = arith.ConstantOp.from_int_and_width(0, i32)
        one = arith.ConstantOp.from_int_and_width(1, i32)
        while_loop = scf.WhileOp(
            [absolute_value, digit_init],
            [i32, i32],
            [before_block],
            [after_block],
        )
        with ImplicitBuilder(before_block) as (running_integer, digits):
            # Stop when you reach zero
            is_zero = arith.CmpiOp(running_integer, zero, "ne")
            scf.ConditionOp(is_zero, running_integer, digits)
        with ImplicitBuilder(after_block) as (running_integer, previous_digits):
            ten = arith.ConstantOp.from_int_and_width(10, 32)
            new_integer = arith.DivUIOp(running_integer, ten)
            digits = arith.AddiOp(previous_digits, one)
            scf.YieldOp(new_integer, digits)

        return while_loop.results[1]

    # print_minus_if_negative implementation
    def print_minus_if_negative(integer: SSAValue) -> SSAValue:
        """
        This function returns an MLIR scf.if op that prints a minus sign
        if the integer value is negative and returns the absolute value
        of the integer in either case, in python code:
        ```
        if(integer > 0):
            print("-")
            return (0 - integer)
        else:
            return integer
        ```
        """
        is_negative_block = Block()
        is_positive_block = Block()
        is_negative = arith.CmpiOp(integer, zero, "slt")

        # Either way get the absolute value of the number
        absolute_value = scf.IfOp(
            is_negative, [i32], [is_negative_block], [is_positive_block]
        )
        # If the value is negative, print the minus sign, return abs value
        with ImplicitBuilder(is_negative_block):
            PrintCharOp.from_constant_char("-")
            negative_input = arith.SubiOp(zero, integer)
            scf.YieldOp(negative_input)
        # If the value is positive, just return value itself as abs value
        with ImplicitBuilder(is_positive_block):
            scf.YieldOp(integer)

        return absolute_value.results[0]

    def print_digits(digits: SSAValue, absolute_value: SSAValue) -> scf.ForOp:
        """
        Return an scf.for loop that prints all digits of the given value.
        In python code:
        for i in range(0, digits, 1):
            position = (digits - 1) - i
            digit = (absolute_value // (10**position)) % 10
            print(digit)
        """
        zero_index = arith.ConstantOp.from_int_and_width(0, IndexType())
        one_index = arith.ConstantOp.from_int_and_width(1, IndexType())
        digits_index = arith.IndexCastOp(digits, IndexType())
        loop_body = Block(arg_types=(IndexType(),))

        # Print all from most significant to least
        for_loop = scf.ForOp(zero_index, digits_index, one_index, (), loop_body)
        with ImplicitBuilder(loop_body) as (index_var,):
            one = arith.ConstantOp.from_int_and_width(1, i32)
            size_minus_one = arith.SubiOp(digits, one)
            index_var_int = arith.IndexCastOp(index_var, i32)
            position = arith.SubiOp(size_minus_one, index_var_int)
            # digit = (num // (10**pos)) % 10
            ten = arith.ConstantOp.from_int_and_width(10, i32)
            i_0 = math.IPowIOp(ten, position)
            i_1 = arith.DivUIOp(absolute_value, i_0)
            digit = arith.RemUIOp(i_1, ten)
            ascii_offset = arith.ConstantOp.from_int_and_width(ord("0"), i32)
            char = arith.AddiOp(digit, ascii_offset)
            char_i8 = arith.TruncIOp(char, i8)
            PrintCharOp(char_i8)
            scf.YieldOp()
        return for_loop

    # Beginning of mlir_itoa declaration
    """
    Python code for this implementation:
    ```
    if(integer = 0):
        print("0")
    else:
        absolute_value = print_minus_if_negative(integer)
        digits = get_number_of_digits(absolute_value)
        print_digits(digits, absolute_value)
    ```
    """
    mlir_itoa_func = func.FuncOp("mlir_itoa", ((i32,), ()))
    with ImplicitBuilder(mlir_itoa_func.body) as (integer,):
        zero = arith.ConstantOp.from_int_and_width(0, i32)
        is_zero_block = Block()
        is_not_zero_block = Block()
        is_zero = arith.CmpiOp(integer, zero, "eq")
        scf.IfOp(
            is_zero,
            [],
            [is_zero_block],
            [is_not_zero_block],
        )
        # If the value is zero, just print one zero
        with ImplicitBuilder(is_zero_block):
            PrintCharOp.from_constant_char("0")
            scf.YieldOp()
        # Otherwise continue with itoa
        with ImplicitBuilder(is_not_zero_block):
            # Print minus sign if negative and return absolute value
            absolute_value = print_minus_if_negative(integer)
            # Get amount  of digits to be printed
            digits = get_number_of_digits(absolute_value)
            print_digits(digits, absolute_value)
            # Yield from is_not_zero_block
            scf.YieldOp()
        # Return from itoa function
        func.ReturnOp()

    return mlir_itoa_func


class PrintfToPutcharPass(ModulePass):
    name = "printf-to-putchar"

    # lower to func.call
    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # Check if there are any printints
        contains_printint = any(
            isinstance(op_in_module, PrintIntOp) for op_in_module in op.walk()
        )
        if contains_printint:
            print_int_walker = PatternRewriteWalker(
                ConvertPrintIntToItoa(),
                apply_recursively=False,
            )
            print_int_walker.rewrite_module(op)
            # This has to happen first, since mlir_itoa
            # contains some references to printf.print_char
            # Add function implementation for mlir_itoa if it isn't there already
            SymbolTable.insert_or_update(op, get_mlir_itoa())
        # Check if there are any printchars
        contains_printchar = any(
            isinstance(op_in_module, PrintCharOp) for op_in_module in op.walk()
        )
        if contains_printchar:
            print_char_walker = PatternRewriteWalker(
                LowerPrintCharToPutchar(),
                apply_recursively=False,
            )
            print_char_walker.rewrite_module(op)
            # Add external putchar reference if not already there
            func_op = func.FuncOp.external("putchar", [i32], [i32])
            SymbolTable.insert_or_update(op, func_op)
