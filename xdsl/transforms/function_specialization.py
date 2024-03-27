from collections.abc import Iterable
from typing import cast

from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Attribute, Block, MLContext, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

SPEZIALIZE_ON_VALS_ATTR = "specialize_on_vals"


class FunctionSpecializationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter, /):
        # check if the function contains a "specialize_on_vals" annotated operation
        split_op = func_contains_specialization_annotation(func_op)
        if split_op is None:
            return

        # can't do splits on multi-value ops
        if len(split_op.results) != 1:
            return

        # get list of vals to specialize on:
        specialization_vals = get_op_specialization(split_op)
        # drop malformed or empty specializations:
        if specialization_vals is None or len(specialization_vals) == 0:
            split_op.attributes.pop(SPEZIALIZE_ON_VALS_ATTR)
            return

        return_types = tuple(func_op.function_type.outputs)

        function_remainder = split_op.next_op
        assert function_remainder is not None, "Can't specialize on terminator!"

        # grab the first value to specialize on:
        val = specialization_vals.pop()
        # generate the specialized function:
        new_func = specialize_function(func_op, val, rewriter)
        # insert the specialized function after the generic function (the one we matched on)
        rewriter.insert_op_after_matched_op(new_func)
        # insert a compare to the value we specialize and, and branch on if we are equal
        rewriter.insert_op_after(
            [
                cst := arith.Constant(val, split_op.results[0].type),
                is_eq := arith.Cmpi(split_op.results[0], cst, "eq"),
                scf_if := scf.If(
                    is_eq,
                    return_types,
                    [
                        # if we are equal to the specialized value, call the function:
                        call_op := func.Call(
                            new_func.sym_name.data,
                            func_op.body.block.args,
                            return_types,
                        ),
                        # yield call results
                        scf.Yield(*call_op.results),
                    ],
                    # empty region placeholder, will be filled in later
                    # grab a reference to it
                    Region(dest_block := Block()),
                ),
            ],
            split_op,
        )

        # iterate over the remainder of the function:
        # grab a reference to the next operation in the remainder.
        # this is because we will modify the op and therefore loose the "old" next op.
        next_op = function_remainder.next_op
        # unless we already hit the block terminator
        if next_op is not None:
            # while we haven't reached the return statement:
            while function_remainder is not func_op.body.block.last_op:
                # detatch the function
                function_remainder.detach()
                # re-insert it inside the else block of the if statement
                rewriter.insert_op_at_end(function_remainder, dest_block)
                # go to next op
                function_remainder = next_op
                next_op = function_remainder.next_op

        # insert a yield that yields the return values
        rewriter.insert_op_at_end(scf.Yield(*function_remainder.operands), dest_block)
        # return the results of the scf.if
        rewriter.replace_op(function_remainder, func.Return(*scf_if.results))

        # remove specialization attribute
        if specialization_vals:
            split_op.attributes[SPEZIALIZE_ON_VALS_ATTR] = ArrayAttr(
                specialization_vals
            )
        else:
            split_op.attributes.pop(SPEZIALIZE_ON_VALS_ATTR)


def specialize_function(
    func_op: func.FuncOp,
    specialization: Attribute,
    rewriter: PatternRewriter,
):
    """
    Specializes a function to pin a value to a compile time constant. Assumes the function is top-level
    inside the module.

    This will do the following things:
    - clone the function
    - rename it to be uniquely named inside the module
    - erase all operations up until the specialized operation
    - replace the specialized operation with a constant instantiation
    """
    # clone the function including the body:
    new_func = func_op.clone()
    # get the module op
    module = func_op.parent_op()
    assert isinstance(module, builtin.ModuleOp), "func must be top-level functions!"
    # generate a new name and set it:
    new_func.sym_name = StringAttr(
        unique_specialized_name(module, new_func.sym_name.data, "specialized")
    )

    # find the first operation that is structurally equivalent, this will always give us the exact same operation
    # that was matched, simply because the function `func_contains_specialization_annotation` returns
    # the first occurrence of any operation with the attribute.
    for op in new_func.body.ops:
        # replace specialized op by constant
        if SPEZIALIZE_ON_VALS_ATTR in op.attributes:
            # find ops that came before, so we can erase them
            for bad_ops in ops_between_op_and_func_start(func_op, op):
                rewriter.erase_op(bad_ops)
            # then check that we really just have one result (sanity check)
            assert (
                len(op.results) == 1
            ), "Specializations only work on single return operations"
            # replace op by constant
            rewriter.replace_op(op, arith.Constant(specialization, op.results[0].type))
            # don't look at more operations inside the function
            break
    # return the newly created func op
    return new_func


def func_contains_specialization_annotation(funcop: func.FuncOp) -> Operation | None:
    """
    Return the first operation inside the function that has a "specialize_on_vals" attribute.

    Only works on top-level operations, we can't handle nested things right now.
    """
    for op in funcop.body.block.ops:
        if SPEZIALIZE_ON_VALS_ATTR in op.attributes:
            return op


def get_op_specialization(op: Operation) -> list[Attribute] | None:
    """
    Reads the "specialize_on_vals" attribute of an operation, checks for valid
    formatting, and return the list of attributes that should be specialized on.
    """
    specialize_attr = op.attributes.get(SPEZIALIZE_ON_VALS_ATTR)
    if not specialize_attr:
        return None
    if not isinstance(specialize_attr, ArrayAttr):
        return None

    return list(cast(ArrayAttr[Attribute], specialize_attr))


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


def unique_specialized_name(module: builtin.ModuleOp, name: str, hint: str) -> str:
    """
    Generate a new specialized name that is unique to the module
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


class FunctionSpecializationPass(ModulePass):
    name = "function-specialization"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FunctionSpecializationPattern()).rewrite_module(op)
