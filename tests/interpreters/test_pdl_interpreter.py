from io import StringIO

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, func, pdl, test
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
from xdsl.interpreters.pdl import (
    PDLMatcher,
    PDLRewriteFunctions,
    PDLRewritePattern,
)
from xdsl.ir import Attribute, OpResult
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.test_value import TestSSAValue


class SwapInputs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
        if not isinstance(op.lhs, OpResult):
            return
        if not isinstance(op.lhs.op, arith.Addi):
            return
        new_op = arith.Addi(op.rhs, op.lhs)
        rewriter.replace_op(op, [new_op])


def test_rewrite_swap_inputs_python():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()

    PatternRewriteWalker(SwapInputs(), apply_recursively=False).rewrite_module(
        input_module
    )

    assert input_module.is_structurally_equivalent(output_module)


def test_rewrite_swap_inputs_pdl():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()
    rewrite_module = swap_arguments_pdl()

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    stream = StringIO()

    ctx = MLContext()
    ctx.load_dialect(arith.Arith)

    PatternRewriteWalker(
        PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
        apply_recursively=False,
    ).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)


def swap_arguments_input():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            z = arith.Constant.from_int_and_width(1, 32).result
            x_y = arith.Addi(x, y).result
            x_y_z = arith.Addi(x_y, z).result
            func.Return(x_y_z)

    return ir_module


def swap_arguments_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32).result
            y = arith.Constant.from_int_and_width(2, 32).result
            z = arith.Constant.from_int_and_width(1, 32).result
            x_y = arith.Addi(x, y).result
            z_x_y = arith.Addi(z, x_y).result
            func.Return(z_x_y)

    return ir_module


def swap_arguments_pdl():
    # The rewrite below matches the second addition as root op

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            x = pdl.OperandOp().value
            y = pdl.OperandOp().value
            pdl_type = pdl.TypeOp().result

            x_y_op = pdl.OperationOp(
                StringAttr("arith.addi"), operand_values=[x, y], type_values=[pdl_type]
            ).op
            x_y = pdl.ResultOp(IntegerAttr(0, 32), parent=x_y_op).val
            z = pdl.OperandOp().value
            overflow = pdl.AttributeOp().output
            x_y_z_op = pdl.OperationOp(
                op_name=StringAttr("arith.addi"),
                operand_values=[x_y, z],
                type_values=[pdl_type],
                attribute_value_names=(StringAttr("overflowFlags"),),
                attribute_values=(overflow,),
            ).op

            with ImplicitBuilder(pdl.RewriteOp(x_y_z_op).body):
                z_x_y_op = pdl.OperationOp(
                    StringAttr("arith.addi"),
                    operand_values=[z, x_y],
                    type_values=[pdl_type],
                    attribute_value_names=(StringAttr("overflowFlags"),),
                    attribute_values=(overflow,),
                ).op
                pdl.ReplaceOp(x_y_z_op, z_x_y_op)

    return pdl_module


class AddZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
        if not isinstance(op.rhs, OpResult):
            return
        if not isinstance(op.rhs.op, arith.Constant):
            return
        rhs = op.rhs.op
        if not isinstance(rhs_value := rhs.value, IntegerAttr):
            return
        if rhs_value.value.data != 0:
            return
        rewriter.replace_matched_op([], new_results=[op.lhs])


def test_rewrite_add_zero_python():
    input_module = add_zero_input()
    output_module = add_zero_output()

    PatternRewriteWalker(AddZero(), apply_recursively=False).rewrite_module(
        input_module
    )

    assert input_module.is_structurally_equivalent(output_module)


def test_rewrite_add_zero_pdl():
    input_module = add_zero_input()
    output_module = add_zero_output()
    rewrite_module = add_zero_pdl()
    # input_module.verify()
    # output_module.verify()
    rewrite_module.verify()

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    stream = StringIO()

    ctx = MLContext()
    ctx.load_dialect(arith.Arith)

    PatternRewriteWalker(
        PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
        apply_recursively=False,
    ).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)


def add_zero_input():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32)
            y = arith.Constant.from_int_and_width(0, 32)
            z = arith.Addi(x, y)
            func.Return(z)

    return ir_module


def add_zero_output():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        with ImplicitBuilder(func.FuncOp("impl", ((), ())).body):
            x = arith.Constant.from_int_and_width(4, 32)
            _y = arith.Constant.from_int_and_width(0, 32)
            func.Return(x)

    return ir_module


def add_zero_pdl():
    # The rewrite below matches the second addition as root op
    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            # Type i32
            pdl_i32 = pdl.TypeOp().result

            # LHS: i32
            lhs = pdl.OperandOp().results[0]

            # Constant 0: i32
            zero = pdl.AttributeOp(value=IntegerAttr(0, 32)).results[0]
            rhs_op = pdl.OperationOp(
                op_name=StringAttr("arith.constant"),
                attribute_value_names=ArrayAttr([StringAttr("value")]),
                attribute_values=[zero],
                type_values=[pdl_i32],
            ).op
            rhs = pdl.ResultOp(IntegerAttr(0, 32), parent=rhs_op).val

            # LHS + 0
            sum = pdl.OperationOp(
                StringAttr("arith.addi"),
                operand_values=[lhs, rhs],
                type_values=[pdl_i32],
            ).op

            with ImplicitBuilder(pdl.RewriteOp(sum).body):
                pdl.ReplaceOp(sum, repl_values=[lhs])

    return pdl_module


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


def constant_zero():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        arith.Constant.from_int_and_width(0, 32)

    return ir_module


def constant_one():
    @ModuleOp
    @Builder.implicit_region
    def ir_module():
        arith.Constant.from_int_and_width(1, 32)

    return ir_module


def change_constant_value_pdl():
    # The rewrite below changes the predicate of a cmpi operation
    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            # Type i32
            pdl_i32 = pdl.TypeOp().result

            # Constant 0: i32
            zero = pdl.AttributeOp(value=IntegerAttr(0, 32)).results[0]
            const_op = pdl.OperationOp(
                op_name=StringAttr("arith.constant"),
                attribute_value_names=ArrayAttr([StringAttr("value")]),
                attribute_values=[zero],
                type_values=[pdl_i32],
            ).op

            with ImplicitBuilder(pdl.RewriteOp(const_op).body):
                # changing constants value via attributes
                one = pdl.AttributeOp(value=IntegerAttr(1, 32)).results[0]
                const_new = pdl.OperationOp(
                    op_name=StringAttr("arith.constant"),
                    attribute_value_names=ArrayAttr([StringAttr("value")]),
                    attribute_values=[one],
                    type_values=[pdl_i32],
                ).op
                pdl.ReplaceOp(const_op, repl_operation=const_new)

    return pdl_module


def test_interpreter_attribute_rewrite():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLRewriteFunctions(MLContext()))

    input_module = constant_zero()
    expected_module = constant_one()
    rewrite_module = change_constant_value_pdl()
    rewrite_module.verify()

    pdl_rewrite_op = next(
        op for op in rewrite_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    stream = StringIO()

    ctx = MLContext()
    ctx.load_dialect(arith.Arith)

    PatternRewriteWalker(
        PDLRewritePattern(pdl_rewrite_op, ctx, file=stream),
        apply_recursively=False,
    ).rewrite_module(input_module)

    assert expected_module.is_structurally_equivalent(input_module)


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


def test_distinct_types_polymorph_matching():
    @ModuleOp
    @Builder.implicit_region
    def input_module():
        test.TestOp.create(result_types=[IntegerType(32)])
        test.TestOp.create(result_types=[IntegerType(64)])

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            pdl_type = pdl.TypeOp().result
            pdl_op = pdl.OperationOp(StringAttr("test.op"), type_values=[pdl_type]).op
            with ImplicitBuilder(pdl.RewriteOp(pdl_op).body):
                pdl.EraseOp(pdl_op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    ctx = MLContext()
    pattern_walker = PatternRewriteWalker(PDLRewritePattern(pdl_rewrite_op, ctx))

    pattern_walker.rewrite_module(input_module)
    assert input_module.is_structurally_equivalent(ModuleOp([]))


def test_distinct_types_matching():
    @ModuleOp
    @Builder.implicit_region
    def input_module():
        test.TestOp.create(result_types=[IntegerType(32)])
        test.TestOp.create(result_types=[IntegerType(64)])

    @ModuleOp
    @Builder.implicit_region
    def output_module():
        test.TestOp.create(result_types=[IntegerType(64)])

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(2, None).body):
            pdl_type = pdl.TypeOp(IntegerType(32)).result
            pdl_op = pdl.OperationOp(StringAttr("test.op"), type_values=[pdl_type]).op
            with ImplicitBuilder(pdl.RewriteOp(pdl_op).body):
                pdl.EraseOp(pdl_op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    ctx = MLContext()
    pattern_walker = PatternRewriteWalker(PDLRewritePattern(pdl_rewrite_op, ctx))

    pattern_walker.rewrite_module(input_module)
    assert input_module.is_structurally_equivalent(output_module)


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
