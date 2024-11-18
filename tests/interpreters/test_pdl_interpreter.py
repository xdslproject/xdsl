from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, pdl, test
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.pdl import PDLMatcher, PDLRewriteFunctions, PDLRewritePattern
from xdsl.ir import Attribute, Block
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.utils.test_value import TestSSAValue


def test_interpreter_functions():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLRewriteFunctions(MLContext()))

    c0 = TestSSAValue(i32)
    c1 = TestSSAValue(i32)
    add = arith.Addi(c0, c1)
    add_res = add.result

    assert interpreter.run_op(
        pdl.ResultOp(0, TestSSAValue(pdl.OperationType())), (add,)
    ) == (add_res,)


def test_native_constraint():
    @ModuleOp
    @Builder.implicit_region
    def input_module_true():
        test.TestOp.create(properties={"attr": StringAttr("foo")})

    @ModuleOp
    @Builder.implicit_region
    def input_module_false():
        test.TestOp.create(properties={"attr": StringAttr("baar")})

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(42, None).body):
            attr = pdl.AttributeOp().output
            pdl.ApplyNativeConstraintOp("even_length_string", [attr])
            op = pdl.OperationOp(
                op_name=None,
                attribute_value_names=ArrayAttr([StringAttr("attr")]),
                attribute_values=[attr],
            ).op
            with ImplicitBuilder(pdl.RewriteOp(op).body):
                pdl.EraseOp(op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    def even_length_string(attr: Attribute) -> bool:
        return isinstance(attr, StringAttr) and len(attr.data) == 4

    ctx = MLContext()
    PDLMatcher.native_constraints["even_length_string"] = even_length_string

    pattern_walker = PatternRewriteWalker(PDLRewritePattern(pdl_rewrite_op, ctx))

    new_input_module_true = input_module_true.clone()
    pattern_walker.rewrite_module(new_input_module_true)

    new_input_module_false = input_module_false.clone()
    pattern_walker.rewrite_module(new_input_module_false)

    assert new_input_module_false.is_structurally_equivalent(ModuleOp([]))
    assert new_input_module_true.is_structurally_equivalent(input_module_true)


def test_match_type():
    matcher = PDLMatcher()

    pdl_op = pdl.TypeOp()
    ssa_value = pdl_op.result
    xdsl_value = StringAttr("a")

    # New value
    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    assert not matcher.match_type(ssa_value, pdl_op, StringAttr("b"))
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_fixed_type():
    matcher = PDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(32))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result

    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_not_match_fixed_type():
    matcher = PDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(64))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result

    assert not matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {}


def test_match_operand():
    matcher = PDLMatcher()

    pdl_op = pdl.OperandOp()
    ssa_value = pdl_op.value
    xdsl_value = TestSSAValue(i32)

    # New value
    assert matcher.match_operand(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_operand(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    other_value = TestSSAValue(i32)
    assert not matcher.match_operand(ssa_value, pdl_op, other_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Wrong type
    type_op = pdl.TypeOp(i64)
    new_pdl_op = pdl.OperandOp(type_op.result)
    new_value = TestSSAValue(i32)
    assert not matcher.match_operand(new_pdl_op.value, new_pdl_op, new_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_result():
    matcher = PDLMatcher()

    type_op = pdl.TypeOp(IntegerType(32))
    operation_op = pdl.OperationOp(op_name=None, type_values=(type_op.result,))
    result_op = pdl.ResultOp(0, operation_op.op)
    xdsl_op = test.TestOp(result_types=(i32,))
    xdsl_value = xdsl_op.res[0]

    # New result
    # If the result of an operation has the expected type we should match
    assert matcher.match_result(result_op.val, result_op, xdsl_value)
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Same result
    # We should accept the same value given the same constraint
    assert matcher.match_result(result_op.val, result_op, xdsl_value)
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Other result
    # We should not match again with a different value, even if it has the correct type
    other_xdsl_op = test.TestOp(result_types=(i32,))
    other_xdsl_value = other_xdsl_op.res[0]

    assert not matcher.match_result(result_op.val, result_op, other_xdsl_value)
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Wrong type
    # Matching should fail if the result's type differs from the expected type
    wrong_type_op = pdl.TypeOp(i64)
    wrong_type_operation_op = pdl.OperationOp(
        op_name=None, type_values=(wrong_type_op.result,)
    )
    wrong_type_result_op = pdl.ResultOp(0, wrong_type_operation_op.op)

    assert not matcher.match_result(
        wrong_type_result_op.val, wrong_type_result_op, xdsl_value
    )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Index out of range
    # If the operation has only one result, we should not match results at different
    # indices
    out_of_range_result_op = pdl.ResultOp(1, operation_op.op)
    assert not matcher.match_result(
        out_of_range_result_op.val, out_of_range_result_op, xdsl_value
    )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Block argument
    # Result patterns should not match on block arguments
    block = Block(arg_types=(i32,))
    block_arg_result_op = pdl.ResultOp(1, operation_op.op)
    assert not matcher.match_result(
        block_arg_result_op.val, block_arg_result_op, block.args[0]
    )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }


def test_native_constraint_constant_parameter():
    """
    Check that `pdl.apply_native_constraint` can take constant attribute parameters
    that are not otherwise matched.
    """

    @ModuleOp
    @Builder.implicit_region
    def input_module_true():
        test.TestOp.create(properties={"attr": StringAttr("foo")})

    @ModuleOp
    @Builder.implicit_region
    def input_module_false():
        test.TestOp.create(properties={"attr": StringAttr("baar")})

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(42, None).body):
            attr = pdl.AttributeOp().output
            four = pdl.AttributeOp(IntegerAttr(4, i32)).output
            pdl.ApplyNativeConstraintOp("length_string", [attr, four])
            op = pdl.OperationOp(
                op_name=None,
                attribute_value_names=ArrayAttr([StringAttr("attr")]),
                attribute_values=[attr],
            ).op
            with ImplicitBuilder(pdl.RewriteOp(op).body):
                pdl.EraseOp(op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    def length_string(attr: Attribute, size: Attribute) -> bool:
        return (
            isinstance(attr, StringAttr)
            and isinstance(size, IntegerAttr)
            and len(attr.data) == size.value.data
        )

    ctx = MLContext()
    PDLMatcher.native_constraints["length_string"] = length_string

    pattern_walker = PatternRewriteWalker(PDLRewritePattern(pdl_rewrite_op, ctx))

    new_input_module_true = input_module_true.clone()
    pattern_walker.rewrite_module(new_input_module_true)

    new_input_module_false = input_module_false.clone()
    pattern_walker.rewrite_module(new_input_module_false)

    assert new_input_module_false.is_structurally_equivalent(ModuleOp([]))
    assert new_input_module_true.is_structurally_equivalent(input_module_true)
