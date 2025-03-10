from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any, cast

from xdsl.context import MLContext
from xdsl.dialects import eqsat, pdl
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, impl, register_impls
from xdsl.interpreters.pdl import PDLMatcher, PDLRewriteFunctions
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint
from xdsl.transforms.convert_onnx_to_linalg import get_root_op
from xdsl.utils.exceptions import InterpretationError


@dataclass
class EqsatPDLMatcher(PDLMatcher):
    def match_operand(
        self, ssa_val: SSAValue, pdl_op: pdl.OperandOp, xdsl_val: SSAValue
    ):
        owner = xdsl_val.owner
        assert isinstance(owner, eqsat.EClassOp)
        assert len(owner.operands) == 1, (
            "newly converted eqsat always has 1 element in eclass"
        )
        arg = owner.operands[0]
        res = super().match_operand(ssa_val, pdl_op, arg)
        # self.value_to_eclass[arg] = owner
        return res

        # res = super().match_operand(ssa_val, pdl_op, xdsl_val)
        # uses = xdsl_val.uses
        # assert len(uses) == 1, f"Eclass representation of code, uses: {uses}"
        # only_use = next(iter(uses))
        # assert isinstance(only_use.operation, eqsat.EClassOp)
        # self.value_to_eclass[ssa_val] = only_use.operation
        # return res


def _get_root_op(op: Operation | None) -> Operation | None:
    """
    Recursively finds and returns the root operation associated with the given operation.
    """
    return op if op is None or op.parent_op() is None else get_root_op(op.parent_op())


@dataclass
class EqsatPDLRewritePattern(RewritePattern):
    functions: EqsatPDLRewriteFunctions
    pdl_rewrite_op: pdl.RewriteOp
    interpreter: Interpreter

    def __init__(
        self, pdl_rewrite_op: pdl.RewriteOp, ctx: MLContext, file: IO[str] | None = None
    ):
        pdl_pattern = pdl_rewrite_op.parent_op()
        assert isinstance(pdl_pattern, pdl.PatternOp)
        pdl_module = pdl_pattern.parent_op()
        assert isinstance(pdl_module, ModuleOp)
        self.functions = EqsatPDLRewriteFunctions(ctx)
        self.interpreter = Interpreter(pdl_module, file=file)
        self.interpreter.register_implementations(self.functions)
        self.pdl_rewrite_op = pdl_rewrite_op

    def match_and_rewrite(self, xdsl_op: Operation, rewriter: PatternRewriter) -> None:
        if not self.functions.did_populate:
            self.functions.populate_maps(cast(ModuleOp, _get_root_op(xdsl_op)))

        pdl_op_val = self.pdl_rewrite_op.root
        assert pdl_op_val is not None, "TODO: handle None root op in pdl.RewriteOp"
        assert (
            self.pdl_rewrite_op.body is not None
        ), "TODO: handle None body op in pdl.RewriteOp"

        assert isinstance(pdl_op_val, OpResult)
        pdl_op = pdl_op_val.op

        assert isinstance(pdl_op, pdl.OperationOp)
        matcher = EqsatPDLMatcher()
        if not matcher.match_operation(pdl_op_val, pdl_op, xdsl_op):
            return

        parent = self.pdl_rewrite_op.parent_op()
        assert isinstance(parent, pdl.PatternOp)
        for constraint_op in parent.walk():
            if isinstance(constraint_op, pdl.ApplyNativeConstraintOp):
                if not matcher.check_native_constraints(constraint_op):
                    return

        self.interpreter.push_scope("rewrite")
        self.interpreter.set_values(matcher.matching_context.items())
        self.functions.rewriter = rewriter
        # self.functions.value_to_eclass = matcher.value_to_eclass

        self.interpreter.run_ssacfg_region(self.pdl_rewrite_op.body, ())

        self.interpreter.pop_scope()
        # Reset values just in case
        # self.functions.value_to_eclass = {}


@register_impls
@dataclass
class EqsatPDLRewriteFunctions(PDLRewriteFunctions):
    """
    The implementations in this class are for the RHS of the rewrite. The SSA values
    referenced within the rewrite block are guaranteed to have been matched with the
    corresponding IR elements. The interpreter context stores the IR elements by SSA
    values.
    """

    value_to_eclass: dict[SSAValue, eqsat.EClassOp] = field(default_factory=dict)
    op_components_to_op: dict[
        tuple[str, tuple[tuple[str, Attribute], ...], tuple[SSAValue, ...]],
        Operation,
    ] = field(default_factory=dict)
    did_populate: bool = field(default=False)

    def populate_maps(self, module: ModuleOp):
        for op in module.walk():
            if isinstance(op, eqsat.EClassOp):
                for operand in op.operands:
                    self.value_to_eclass[operand] = op
                    if isinstance(operand, OpResult):
                        source = operand.op
                        attributes = tuple(
                            sorted(source.attributes.items(), key=lambda bla: bla[0])
                        )
                        self.op_components_to_op[
                            (source.name, attributes, tuple(source.operands))
                        ] = source
        self.did_populate = True

    @impl(pdl.OperationOp)
    def run_operation(
        self, interpreter: Interpreter, op: pdl.OperationOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert op.opName is not None
        op_name = op.opName.data
        op_type = self.ctx.get_optional_op(op_name)

        if op_type is None:
            raise InterpretationError(
                f"Could not find op type for name {op_name} in context"
            )

        attribute_value_names = [avn.data for avn in op.attributeValueNames.data]

        # How to deal with operandSegmentSizes?
        # operand_values, attribute_values, type_values = args

        operand_values = interpreter.get_values(op.operand_values)
        for operand in operand_values:
            assert isinstance(operand, SSAValue)
            assert len(operand.uses) == 1

        operand_eqsat_values = tuple(
            next(iter(ov.uses)).operation.results[0]
            for ov in cast(tuple[SSAValue, ...], operand_values)
        )

        attribute_values = interpreter.get_values(op.attribute_values)

        for attribute in attribute_values:
            assert isinstance(attribute, Attribute)

        type_values = interpreter.get_values(op.type_values)

        for type_value in type_values:
            assert isinstance(type_value, TypeAttribute)

        attributes = dict[str, Attribute]()
        properties = dict[str, Attribute]()

        # If the op is an IRDL-defined operation, get the property names.
        if issubclass(op_type, IRDLOperation):
            property_names = op_type.get_irdl_definition().properties.keys()
        else:
            property_names = []

        # Move the attributes to the attribute or property dictionary
        # depending on whether they are a properties or not.
        for attribute_name, attribute_value in zip(
            attribute_value_names, attribute_values
        ):
            if attribute_name in property_names:
                properties[attribute_name] = attribute_value
            else:
                attributes[attribute_name] = attribute_value

        # check if already exists
        attributes_key = tuple(sorted(attributes.items(), key=lambda bla: bla[0]))
        key = (op_name, attributes_key, operand_eqsat_values)
        existing_op = self.op_components_to_op.get(key)
        if existing_op is not None:
            return (existing_op,)

        eclass_operand_values = tuple(
            self.value_to_eclass[operand].result for operand in operand_values
        )

        result_op = op_type.create(
            operands=eclass_operand_values,
            result_types=type_values,
            attributes=attributes,
            properties=properties,
        )

        self.op_components_to_op[key] = result_op

        return (result_op,)

    @impl(pdl.ReplaceOp)
    def run_replace(
        self, interpreter: Interpreter, op: pdl.ReplaceOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        rewriter = self.rewriter

        (old_op,) = interpreter.get_values((op.op_value,))
        assert isinstance(old_op, Operation)
        old_eclass_op = self.value_to_eclass[old_op.results[0]]

        assert not op.repl_values

        assert op.repl_operation is not None or op.repl_values

        if op.repl_operation is not None:
            (new_op,) = interpreter.get_values((op.repl_operation,))
            assert isinstance(new_op, Operation)
            new_results = new_op.results
            assert len(new_results) == 1
            new_eclass_op = self.value_to_eclass.get(new_results[0])

            if new_eclass_op is None:
                # Add new op to eclass
                new_results[0].name_hint = old_op.results[0].name_hint
                rewriter.insert_op(new_op, InsertPoint.before(old_op))
                old_eclass_op.operands = old_eclass_op.arguments + new_results
                self.value_to_eclass[new_results[0]] = old_eclass_op
            else:
                if old_eclass_op is new_eclass_op:
                    # Class already exists and is in the same eclass, nothing to do
                    ...
                else:
                    raise NotImplementedError()

        elif op.repl_values:
            raise NotImplementedError()

        return ()
