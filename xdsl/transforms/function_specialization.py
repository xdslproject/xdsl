from typing import cast, Iterable

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Operation, Attribute, MLContext, Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
)

from xdsl.dialects import func, arith, builtin, scf
from xdsl.traits import SymbolTable


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

        if len(specialization_vals) > 1:
            print("can't do more than one specialization yet!")
            return

        return_types = tuple(func_op.function_type.outputs)

        function_remainder = split_op.next_op

        # for every specialization, generate a specialized function body:
        dest_block: Block | None = None
        for val in specialization_vals:
            new_func = specialize_function(func_op, (split_op, val), rewriter)
            rewriter.insert_op_after_matched_op(new_func)
            rewriter.insert_op_after(
                [
                    cst := arith.Constant(val, split_op.results[0].type),
                    is_eq := arith.Cmpi(split_op.results[0], cst, "eq"),
                    scf_if := scf.If(
                        is_eq,
                        return_types,
                        [
                            call_op := func.Call(
                                new_func.sym_name.data, func_op.body.block.args, return_types
                            ),
                            scf.Yield(*call_op.results),
                        ],
                        Region(Block()),
                    ),
                ],
                split_op,
            )
            dest_block = scf_if.false_region.block

        if dest_block is not None:
            next = function_remainder.next_op
            while function_remainder is not func_op.body.block.last_op:
                function_remainder.detach()
                rewriter.insert_op_at_end(function_remainder, dest_block)
                function_remainder = next
                next = function_remainder.next_op

            rewriter.insert_op_at_end(scf.Yield(*function_remainder.operands), dest_block)

            rewriter.replace_op(function_remainder, func.Return(*dest_block.parent_op().results))

        # remove specialization attribute
        split_op.attributes.pop("specialize_on_vals")


def specialize_function(
    func_op: func.FuncOp,
    specialization: tuple[Operation, Attribute],
    rewriter: PatternRewriter,
):
    """ """
    new_func = func_op.clone()
    # generate a new name and set it:
    module = func_op.parent_op()
    assert isinstance(module, builtin.ModuleOp), "func must be top-level functions!"
    new_func.sym_name = StringAttr(
        unique_specialized_name(module, new_func.sym_name.data, "specialized")
    )

    # find the first operation that is structurally equivalent, this will always give us the exact same operation
    # that was matched, simply because the function `func_contains_specialization_annotation` returns
    # the first occurrence of any operation with the attribute.
    for op in new_func.body.ops:
        # replace specialized op by constant
        if "specialize_on_vals" in op.attributes:
            # find ops that came before, so we can erase them
            for bad_ops in ops_between_op_and_func_start(func_op, op):
                rewriter.erase_op(bad_ops)
            # then check that we really just have one result (sanity check)
            assert (
                len(op.results) == 1
            ), "Specializations only work on single return operations"
            # replace op by constant
            rewriter.replace_op(
                op, arith.Constant(specialization[1], op.results[0].type)
            )
            break
    # return the newly created func op
    return new_func


def func_contains_specialization_annotation(funcop: func.FuncOp) -> Operation:
    """
    Return the first operation inside the function that has a "specialize_on_vals" attribute.
    """
    for op in funcop.body.block.ops:
        if "specialize_on_vals" in op.attributes:
            return op


def get_op_specialization(op: Operation) -> list[Attribute] | None:
    """
    Reads the "specialize_on_vals" attribute of an operation, checks for valid
    formatting, and return the list of attributes that should be specialized on.
    """
    specialize_attr = op.attributes.get("specialize_on_vals")
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
    Returns them in reverse order of occurence.

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
    Generate a new specialized name that is unqiue to the module
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
