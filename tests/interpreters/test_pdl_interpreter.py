from io import StringIO

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
from xdsl.ir import Attribute
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def
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


@irdl_op_definition
class OnePropOp(IRDLOperation):
    name = "test.one_prop"

    prop = prop_def(IntegerType)


def test_property_rewrite():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLRewriteFunctions(MLContext()))

    @ModuleOp
    @Builder.implicit_region
    def input_i32():
        OnePropOp.create(properties={"prop": i32})

    @ModuleOp
    @Builder.implicit_region
    def input_i64():
        OnePropOp.create(properties={"prop": i64})

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(42, None).body):
            attr_i32 = pdl.AttributeOp(i32).results[0]
            const_op = pdl.OperationOp(
                op_name=OnePropOp.name,
                attribute_value_names=ArrayAttr([StringAttr("prop")]),
                attribute_values=[attr_i32],
            ).op

            with ImplicitBuilder(pdl.RewriteOp(const_op).body):
                # changing constants value via attributes
                attr_i64 = pdl.AttributeOp(i64).results[0]
                const_new = pdl.OperationOp(
                    op_name=StringAttr(OnePropOp.name),
                    attribute_value_names=ArrayAttr([StringAttr("prop")]),
                    attribute_values=[attr_i64],
                ).op
                pdl.ReplaceOp(const_op, repl_operation=const_new)

    input_module = input_i32
    expected_module = input_i64
    rewrite_module = pdl_module
    rewrite_module.verify()

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    stream = StringIO()

    ctx = MLContext()
    ctx.load_dialect(arith.Arith)
    ctx.load_op(OnePropOp)

    PatternRewriteWalker(
        PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
        apply_recursively=False,
    ).rewrite_module(input_module)
    assert str(expected_module) == str(input_module)
    assert expected_module.is_structurally_equivalent(input_module)


def test_erase_op():
    @ModuleOp
    @Builder.implicit_region
    def input_module():
        test.TestOp.create()

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(42, None).body):
            op = pdl.OperationOp(
                op_name=test.TestOp.name,
            ).op
            with ImplicitBuilder(pdl.RewriteOp(op).body):
                pdl.EraseOp(op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    ctx = MLContext()
    pattern_walker = PatternRewriteWalker(PDLRewritePattern(pdl_rewrite_op, ctx))

    pattern_walker.rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(ModuleOp([]))


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
