from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.dialects.irdl import irdl
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.ir.core import Dialect, ParametrizedAttribute
from xdsl.irdl.irdl import AnyAttr, AnyOf, AttrConstraint, EqAttrConstraint, IRDLOperation, OpDef, ParamAttrConstraint, ParamAttrDef, VarConstraint
from xdsl.traits import SymbolTable

def make_dialect(self, dialect: irdl.DialectOp):

        ops : dict[SymbolRefAttr, IRDLOperation] = {}
        attrs : dict[SymbolRefAttr, ParametrizedAttribute] = {}

        for entry in dialect.body.block:
            match entry:
                case irdl.OperationOp():
                    ops.append(self.make_operation(entry))

                case irdl.TypeOp():
                    types.append(self.make_type(entry))



        return Dialect(dialect.sym_name.data, ops, types)

    def make_operation(self, operation:irdl.OperationOp):
        pass

@register_impls
class IRDLFunctions(InterpreterFunctions):

    variable_counter = 0

    def variable_wrap(self, constr:AttrConstraint):
        self.variable_counter += 1
        return VarConstraint(f"V{self.variable_counter}", constr)

    @impl(irdl.IsOp)
    def run_is(self, interpreter: Interpreter, op: irdl.IsOp, args: PythonValues):
        constr = EqAttrConstraint(args[0])
        if len(op.output.uses) > 1:
             constr = self.variable_wrap(constr)
        return (constr,)

    @impl(irdl.AnyOfOp)
    def run_any_of(self, interpreter:Interpreter, op: irdl.AnyOfOp, args: PythonValues):
        constr = AnyOf(args)
        if len(op.output.uses) > 1:
             constr = self.variable_wrap(constr)
        return (constr,)

    @impl(irdl.AnyOp)
    def run_any(self, interpreter:Interpreter, op: irdl.AnyOp, args: PythonValues):
        constr = AnyAttr()
        if len(op.output.uses) > 1:
             constr = self.variable_wrap(constr)
        return (constr,)


@register_impls
class IRDLTypeFunctions()
    @impl(irdl.ParametricOp)
    def run_parameters(self, interpreter:Interpreter, op: irdl.ParametricOp, args: PythonValues):
        base_type = SymbolTable.lookup_symbol(op, op.base_type)

        constr = ParamAttrConstraint(op.base_type, args)
