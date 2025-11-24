from collections.abc import Iterable
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable

PIN_CONSTANT_VALS = "pin_to_constants"


class FunctionConstantPinning(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        # can't rewrite nested functions yet
        if not isinstance(func_op.parent_op(), builtin.ModuleOp):
            return

        # check if the function contains a "pin_to_constants" annotated operation
        split_op = func_contains_pinning_annotation(func_op)
        if split_op is None:
            return
        # can't do splits on multi-value ops
        if len(split_op.results) != 1:
            return

        # get list of vals to pin:
        pinned_vals = get_pinned_vals_for_op(split_op)
        # drop malformed or empty pinning:
        if pinned_vals is None or len(pinned_vals) == 0:
            split_op.attributes.pop(PIN_CONSTANT_VALS)
            return

        return_types = tuple(func_op.function_type.outputs)

        function_remainder = split_op.next_op
        if function_remainder is None:
            # "Can't ping values for terminator"
            return

        # grab the first value to pin:
        val = pinned_vals.pop()
        # generate the function containing pinned value:
        new_func = generate_func_with_pinned_val(func_op, val, rewriter)
        # insert the specialized function after the generic function (the one we matched on)
        rewriter.insert_op(new_func, InsertPoint.after(func_op))
        # insert a compare to the value we specialize and, and branch on if we are equal
        rewriter.insert_op(
            [
                cst := arith.ConstantOp(val, split_op.results[0].type),
                is_eq := arith.CmpiOp(split_op.results[0], cst, "eq"),
                scf_if := scf.IfOp(
                    is_eq,
                    return_types,
                    [
                        # if we are equal to the specialized value, call the function:
                        call_op := func.CallOp(
                            new_func.sym_name.data,
                            func_op.body.block.args,
                            return_types,
                        ),
                        # yield call results
                        scf.YieldOp(*call_op.results),
                    ],
                    # empty region placeholder, will be filled in later
                    # grab a reference to it
                    Region(dest_block := Block()),
                ),
            ],
            InsertPoint.after(split_op),
        )

        # iterate over the remainder of the function:
        # grab a reference to the next operation in the remainder.
        # this is because we will modify the op and therefore loose the "old" next op.
        next_op = function_remainder.next_op

        # unless we already hit the block terminator
        # while we haven't reached the return statement:
        while (
            next_op is not None and function_remainder is not func_op.body.block.last_op
        ):
            # detatch the function
            function_remainder.detach()
            # re-insert it inside the else block of the if statement
            rewriter.insert_op(function_remainder, InsertPoint.at_end(dest_block))
            # go to next op
            function_remainder = next_op
            next_op = function_remainder.next_op

        # insert a yield that yields the return values
        rewriter.insert_op(
            scf.YieldOp(*function_remainder.operands), InsertPoint.at_end(dest_block)
        )
        # return the results of the scf.if
        rewriter.replace_op(function_remainder, func.ReturnOp(*scf_if.results))

        # remove pinning attribute
        if pinned_vals:
            split_op.attributes[PIN_CONSTANT_VALS] = ArrayAttr(pinned_vals)
        else:
            split_op.attributes.pop(PIN_CONSTANT_VALS)


def generate_func_with_pinned_val(
    func_op: func.FuncOp,
    pin: IntegerAttr[IntegerType | IndexType],
    rewriter: PatternRewriter,
):
    """
    Specializes a function to pin a value to a compile time constant. Assumes the
    function is top-level inside the module.

    This will do the following things:
    - clone the function
    - rename it to be uniquely named inside the module
    - erase all operations up until the operation producing the pinned value
    - replace the operation with a constant instantiation
    """
    # clone the function including the body:
    new_func = func_op.clone()
    # get the module op
    module = func_op.parent_op()
    # checked before calling
    assert isinstance(module, builtin.ModuleOp), "func must be top-level functions!"
    # generate a new name and set it:
    new_func.sym_name = StringAttr(
        unique_pinned_name(module, new_func.sym_name.data, "pinned")
    )

    # find the first operation that is structurally equivalent, this will always give us the exact same operation
    # that was matched, simply because the function `func_contains_pinning_annotation` returns
    # the first occurrence of any operation with the attribute.
    for op in new_func.body.ops:
        # replace specialized op by constant
        if PIN_CONSTANT_VALS in op.attributes:
            # find ops that came before, so we can erase them
            for bad_ops in ops_between_op_and_func_start(func_op, op):
                rewriter.erase_op(bad_ops)
            # then check that we really just have one result (sanity check)
            assert len(op.results) == 1, (
                "Constant pinning only work on single return operations"
            )
            # replace op by constant
            rewriter.replace_op(op, arith.ConstantOp(pin, op.results[0].type))
            # don't look at more operations inside the function
            break
    # return the newly created func op
    return new_func


def func_contains_pinning_annotation(funcop: func.FuncOp) -> Operation | None:
    """
    Return the first operation inside the function that has a "pin_to_constants" attribute.

    Only works on top-level operations, we can't handle nested things right now.
    """
    if not funcop.body.blocks:
        return None
    for op in funcop.body.block.ops:
        if PIN_CONSTANT_VALS in op.attributes:
            return op


def get_pinned_vals_for_op(
    op: Operation,
) -> list[IntegerAttr[IntegerType | IndexType]] | None:
    """
    Reads the "pin_to_constants" attribute of an operation, checks for valid
    formatting, and return the list of attribute values that should be pinned.
    """
    pin_attr = op.attributes.get(PIN_CONSTANT_VALS)
    if not pin_attr:
        return None
    if not isinstance(pin_attr, ArrayAttr):
        return None

    return list(cast(ArrayAttr[IntegerAttr[IntegerType | IndexType]], pin_attr))


def ops_between_op_and_func_start(
    func_op: func.FuncOp, op: Operation
) -> Iterable[Operation]:
    """
    Get a list of all operations localed between op and the start of body.
    Returns them in reverse order of occurrence.

    op must be a direct child of func_op!

    func.func @test() { // <- func_op
      test.test()       // A
      test.test()       // B
      test.test()       // <- op
      test.test()       // C

    should return only B, A not C
    """
    # yield all ops before the op, don't yield the op itself
    while op.prev_op is not None:
        op = op.prev_op
        yield op


def unique_pinned_name(module: builtin.ModuleOp, name: str, hint: str) -> str:
    """
    Generate a new name that is unique to the module
    """
    # try just name + hint
    proposed_name = f"{name}_{hint}"
    # prepare a counter if needed
    counter = 1
    # grab symbol table
    iface = module.get_trait(SymbolTable)
    assert iface is not None, "ModuleOp must have symbol table trait!"
    # while name is not unique
    while iface.lookup_symbol(module, proposed_name) is not None:
        # generate new name try
        proposed_name = f"{name}_{hint}_{counter}"
        counter += 1
    # return unique name
    return proposed_name


class FunctionConstantPinningPass(ModulePass):
    """
    This pass consumes IR annotated with special hints to generate new functions that have certain SSA values pinned
    to a constant, usually to enable further optimization options on this pinned function.

    The original function is changed to dynamically dispatch to this pinned function when the ssa value matches the
    given constant.

    Any single-result operation annotated with a "pin_to_constants" attribute containing an array of values, that is
    located within a function body triggers this optimization. These annotations are usually inserted by previous
    passes that know that they would want to generate a more optimized version of their function for specific values
    of a run-time determined variable.

    An example might be a function that branches repeatedly on a specific variable:

    ```
    function test() {
        x = calc_condition()
        if (x) {
          specific_thing()
        }

        some_thing() // A

        if (x) {
          another_thing()
        }

        some_thing() // B
    }
    ```

    if we can pin `x` to `true`, we are suddenly able to generate two much simple function bodies (after constant folding)

    ```
    function test() {
        x = calc_condition()
        if (x) {
            test_pinned()
            return
        }

        some_thing() // A
        some_thing() // B
    }

    function test_pinned() {
        specific_thing()
        some_thing() // A
        another_thing()
        some_thing() // B
    }
    ```

    Note that the function `test_pinned` might be much easier to optimize for a compiler if there are state
    dependencies between `specific_thing`, `another_thing` and `some_thing`.
    """

    name = "function-constant-pinning"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FunctionConstantPinning()).rewrite_module(op)
