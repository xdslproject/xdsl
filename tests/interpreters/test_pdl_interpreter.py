from dataclasses import (dataclass, field)
from io import StringIO
from typing import Any

from xdsl.ir import Attribute, BlockArgument, MLContext, MLIRType, OpResult, Operation, SSAValue
from xdsl.dialects import arith, func, pdl
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr
from xdsl.pattern_rewriter import (PatternRewriter, RewritePattern,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   AnonymousRewritePattern)
from xdsl.interpreter import (Interpreter, InterpreterFunctions,
                              register_impls, impl)
from xdsl.utils.exceptions import InterpretationError


class SwapInputs(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter, /):
        if not isinstance(op.lhs, OpResult):
            return
        if not isinstance(op.lhs.op, arith.Addi):
            return
        new_op = arith.Addi.get(op.rhs, op.lhs)
        rewriter.replace_op(op, [new_op])


def test_rewrite_python():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()

    PatternRewriteWalker(SwapInputs(),
                         apply_recursively=False).rewrite_module(input_module)

    assert input_module.is_structurally_equivalent(output_module)


@dataclass
class PDLMatcher:
    assignment: dict[SSAValue, Operation | Attribute
                     | SSAValue] = field(default_factory=dict)

    def match_operand(self, ssa_val: SSAValue, pdl_op: pdl.OperandOp,
                      xdsl_val: SSAValue):
        if ssa_val in self.assignment:
            return True

        if pdl_op.valueType is not None:
            assert isinstance(pdl_op.valueType, OpResult)
            assert isinstance(pdl_op.valueType.op, pdl.TypeOp)

            if not self.type(pdl_op.valueType, pdl_op.valueType.op,
                             xdsl_val.typ):
                return False

        self.assignment[ssa_val] = xdsl_val

        return True

    def match_result(self, ssa_val: SSAValue, pdl_op: pdl.ResultOp,
                     xdsl_operand: SSAValue):
        if ssa_val in self.assignment:
            return True

        root_pdl_op_value = pdl_op.parent_
        assert isinstance(root_pdl_op_value, OpResult)
        assert isinstance(root_pdl_op_value.op, pdl.OperationOp)

        if not isinstance(xdsl_operand, OpResult):
            return False

        xdsl_op = xdsl_operand.op

        if not self.match_operation(root_pdl_op_value, root_pdl_op_value.op,
                                    xdsl_op):
            return False

        original_op = root_pdl_op_value.op

        index = pdl_op.index.value.data

        if len(original_op.results) <= index:
            return False

        self.assignment[ssa_val] = xdsl_op.results[index]

        return True

    def type(self, ssa_val: SSAValue, pdl_op: pdl.TypeOp,
             xdsl_attr: Attribute):
        if ssa_val in self.assignment:
            # TODO: check type is the same?
            return True

        self.assignment[ssa_val] = xdsl_attr

        return True

    def match_attribute(self, ssa_val: SSAValue, pdl_op: pdl.AttributeOp,
                        attr_name: str, xdsl_attr: Attribute):
        if ssa_val in self.assignment:
            return True

        if pdl_op.value is not None:
            if pdl_op.value != xdsl_attr:
                return False

        if pdl_op.valueType is not None:
            assert isinstance(pdl_op.valueType, OpResult)
            assert isinstance(pdl_op.valueType.op, pdl.TypeOp)

            # TODO: what to do? what does type mean?
            assert False
            # if not PDLFunctionTable.match_pdl_type_op(
            #         interpreter, pdl_op.valueType, pdl_op.valueType.op,
            #         xdsl_attr.typ, assignment):
            #     return False

        self.assignment[ssa_val] = xdsl_attr

        return True

    def match_operation(self, ssa_val: SSAValue, pdl_op: pdl.OperationOp,
                        xdsl_op: Operation) -> bool:
        if ssa_val in self.assignment:
            return True

        if pdl_op.opName is not None:
            if xdsl_op.name != pdl_op.opName.data:
                return False

        attribute_value_names = [
            avn.data for avn in pdl_op.attributeValueNames.data
        ]

        for avn, av in zip(attribute_value_names, pdl_op.attributeValues):
            assert isinstance(av, OpResult)
            assert isinstance(av.op, pdl.AttributeOp)
            if avn not in xdsl_op.attributes:
                return False

            if not self.match_attribute(av, av.op, avn,
                                        xdsl_op.attributes[avn]):
                return False

        pdl_operands = pdl_op.operandValues
        xdsl_operands = xdsl_op.operands

        if len(pdl_operands) != len(xdsl_operands):
            return False

        for pdl_operand, xdsl_operand in zip(pdl_operands, xdsl_operands):
            assert isinstance(pdl_operand, OpResult)
            assert isinstance(pdl_operand.op, pdl.OperandOp | pdl.ResultOp)
            if isinstance(pdl_operand.op, pdl.OperandOp):
                if not self.match_operand(pdl_operand, pdl_operand.op,
                                          xdsl_operand):
                    return False
            elif isinstance(pdl_operand.op, pdl.ResultOp):
                if not self.match_result(pdl_operand, pdl_operand.op,
                                         xdsl_operand):
                    return False

        pdl_results = pdl_op.typeValues
        xdsl_results = xdsl_op.results

        if len(pdl_results) != len(xdsl_results):
            return False

        for pdl_result, xdsl_result in zip(pdl_results, xdsl_results):
            assert isinstance(pdl_result, OpResult)
            assert isinstance(pdl_result.op, pdl.TypeOp)
            if not self.type(pdl_result, pdl_result.op, xdsl_result.typ):
                return False

        self.assignment[ssa_val] = xdsl_op

        return True


@register_impls
@dataclass
class PDLFunctions(InterpreterFunctions):
    ctx: MLContext
    module: ModuleOp
    _rewriter: PatternRewriter | None = field(default=None)

    @property
    def rewriter(self) -> PatternRewriter:
        assert self._rewriter is not None
        return self._rewriter

    @rewriter.setter
    def rewriter(self, rewriter: PatternRewriter):
        self._rewriter = rewriter

    @impl(pdl.PatternOp)
    def run_pattern(self, interpreter: Interpreter, op: pdl.PatternOp,
                    args: tuple[Any, ...]) -> tuple[Any, ...]:
        ops = op.regions[0].ops
        if not len(ops):
            raise InterpretationError('No ops in pattern')

        if not isinstance(ops[-1], pdl.RewriteOp):
            raise InterpretationError('Expected last ')

        for r_op in ops[:-1]:
            # in forward pass, the Python value is the SSA value itself
            if len(r_op.results) != 1:
                raise InterpretationError('PDL ops must have one result')
            result = r_op.results[0]
            interpreter.set_values(((result, r_op), ))

        interpreter.run(ops[-1])

        return ()

    @impl(pdl.RewriteOp)
    def run_rewrite(self, interpreter: Interpreter,
                    pdl_rewrite_op: pdl.RewriteOp,
                    args: tuple[Any, ...]) -> tuple[Any, ...]:

        input_module = self.module

        def rewrite(xdsl_op: Operation, rewriter: PatternRewriter) -> None:

            pdl_op_val = pdl_rewrite_op.root
            assert pdl_op_val is not None, 'TODO: handle None root op in pdl.RewriteOp'
            assert pdl_rewrite_op.body is not None, 'TODO: handle None body op in pdl.RewriteOp'

            pdl_op, = interpreter.get_values((pdl_op_val, ))
            assert isinstance(pdl_op, pdl.OperationOp)
            matcher = PDLMatcher()
            if not matcher.match_operation(pdl_op_val, pdl_op, xdsl_op):
                return

            interpreter.push_scope('rewrite')
            interpreter.set_values(matcher.assignment.items())
            self.rewriter = rewriter

            for rewrite_impl_op in pdl_rewrite_op.body.ops:
                interpreter.run(rewrite_impl_op)

            interpreter.pop_scope()

        rewriter = AnonymousRewritePattern(rewrite)

        PatternRewriteWalker(
            rewriter, apply_recursively=False).rewrite_module(input_module)

        return ()

    @impl(pdl.OperationOp)
    def run_operation(self, interpreter: Interpreter, op: pdl.OperationOp,
                      args: tuple[Any, ...]) -> tuple[Any, ...]:
        assert op.opName is not None
        op_name = op.opName.data
        op_type = self.ctx.get_optional_op(op_name)

        if op_type is None:
            raise InterpretationError(
                f'Could not find op type for name {op_name} in context')

        attribute_value_names = [
            avn.data for avn in op.attributeValueNames.data
        ]

        # How to deal with operand_segment_sizes?
        # operand_values, attribute_values, type_values = args

        operand_values = interpreter.get_values(op.operandValues)
        for operand in operand_values:
            assert isinstance(operand, SSAValue)

        attribute_values = interpreter.get_values(op.attributeValues)

        for attribute in attribute_values:
            assert isinstance(attribute, Attribute)

        type_values = interpreter.get_values(op.typeValues)

        for type_value in type_values:
            assert isinstance(type_value, MLIRType)

        attributes = dict(zip(attribute_value_names, attribute_values))

        result_op = op_type.create(operands=operand_values,
                                   result_types=type_values,
                                   attributes=attributes)

        return result_op,

    @impl(pdl.ReplaceOp)
    def run_replace(self, interpreter: Interpreter, op: pdl.ReplaceOp,
                    args: tuple[Any, ...]) -> tuple[Any, ...]:
        rewriter = self.rewriter

        assert len(args) == 2, "Only handle replace a by b"
        assert isinstance(args[0], Operation)
        assert isinstance(args[1], Operation)

        old, new = args

        rewriter.replace_op(old, new)

        return ()

    @impl(ModuleOp)
    def run_module(self, interpreter: Interpreter, op: ModuleOp,
                   args: tuple[Any, ...]) -> tuple[Any, ...]:
        ops = op.ops
        if len(ops) != 1 or not isinstance(ops[0], pdl.PatternOp):
            raise InterpretationError(
                'Expected single pattern op in pdl module')
        return self.run_pattern(interpreter, ops[0], args)


def test_rewrite_pdl():
    input_module = swap_arguments_input()
    output_module = swap_arguments_output()
    rewrite_module = swap_arguments_pdl()

    stream = StringIO()
    interpreter = Interpreter(rewrite_module, file=stream)
    ctx = MLContext()
    ctx.register_dialect(arith.Arith)

    pdl_ft = PDLFunctions(ctx, input_module)
    interpreter.register_implementations(pdl_ft)

    interpreter.run_module()

    assert input_module.is_structurally_equivalent(output_module)


def swap_arguments_input():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Constant.from_int_and_width(1, 32)
    b3 = arith.Addi.get(b2.result, b1.result)
    b4 = arith.Addi.get(b3.result, b0.result)
    b5 = func.Return.get(b4.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3, b4, b5])

    return ir_module


def swap_arguments_output():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Constant.from_int_and_width(1, 32)
    b3 = arith.Addi.get(b2.result, b1.result)
    b4 = arith.Addi.get(b0.result, b3.result)
    b5 = func.Return.get(b4.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3, b4, b5])

    return ir_module


def swap_arguments_pdl():
    # The rewrite below matches the second addition as root op

    def pattern(*args: BlockArgument) -> list[Operation]:
        p0 = pdl.OperandOp.get()
        p1 = pdl.OperandOp.get()
        p2 = pdl.TypeOp.get()
        p3 = pdl.OperationOp.get(StringAttr("arith.addi"),
                                 operandValues=[p0.results[0], p1.results[0]],
                                 typeValues=[p2.results[0]])
        p4 = pdl.ResultOp.get(IntegerAttr.from_int_and_width(0, 32),
                              parent=p3.results[0])
        p5 = pdl.OperandOp.get()
        p6 = pdl.OperationOp.get(opName=StringAttr("arith.addi"),
                                 operandValues=[p4.results[0], p5.results[0]],
                                 typeValues=[p2.results[0]])

        def rewrite(*args: BlockArgument) -> list[Operation]:
            p7 = pdl.OperationOp.get(
                opName=StringAttr("arith.addi"),
                operandValues=[p5.results[0], p4.results[0]],
                typeValues=[p2.results[0]])
            o0 = pdl.ReplaceOp.get(p6.results[0], p7.results[0])
            return [p7, o0]

        o1 = pdl.RewriteOp.from_callable(None, p6.results[0], [], rewrite)

        return [p0, p1, p2, p3, p4, p5, p6, o1]

    o2 = pdl.PatternOp.from_callable(IntegerAttr.from_int_and_width(2, 16),
                                     None, pattern)

    pdl_module = ModuleOp.from_region_or_ops([o2])

    return pdl_module


def constant_folding_input():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Addi.get(b0.result, b1.result)
    b3 = func.Return.get(b2.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3])

    return ir_module


def constant_folding_output():

    b0 = arith.Constant.from_int_and_width(4, 32)
    b1 = arith.Constant.from_int_and_width(2, 32)
    b2 = arith.Constant.from_int_and_width(6, 32)
    b3 = func.Return.get(b2.result)

    ir_module = ModuleOp.from_region_or_ops([b0, b1, b2, b3])

    return ir_module
