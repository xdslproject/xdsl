from typing import cast

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.irdl import irdl
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.ir import Dialect, ParametrizedAttribute
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    AttrConstraint,
    EqAttrConstraint,
    IRDLOperation,
    OpDef,
    ParamAttrConstraint,
    ParamAttrDef,
    VarConstraint,
    get_accessors_from_op_def,
    get_accessors_from_param_attr_def,
)


@register_impls
class IRDLFunctions(InterpreterFunctions):

    variable_counter = 0

    @staticmethod
    def dialects(interpreter: Interpreter):
        return interpreter.get_data(IRDLFunctions, "irdl.dialects", dict)

    def variable_wrap(self, constr: AttrConstraint):
        self.variable_counter += 1
        return VarConstraint(f"V{self.variable_counter}", constr)

    @impl(irdl.IsOp)
    def run_is(self, interpreter: Interpreter, op: irdl.IsOp, args: PythonValues):
        constr = EqAttrConstraint(op.expected)
        if len(op.output.uses) > 1:
            constr = self.variable_wrap(constr)
        return (constr,)

    @impl(irdl.AnyOfOp)
    def run_any_of(
        self, interpreter: Interpreter, op: irdl.AnyOfOp, args: PythonValues
    ):
        constr = AnyOf(args)
        if len(op.output.uses) > 1:
            constr = self.variable_wrap(constr)
        return (constr,)

    @impl(irdl.AnyOp)
    def run_any(self, interpreter: Interpreter, op: irdl.AnyOp, args: PythonValues):
        constr = AnyAttr()
        if len(op.output.uses) > 1:
            constr = self.variable_wrap(constr)
        return (constr,)

    @impl(irdl.ParametricOp)
    def run_parametric(
        self, interpreter: Interpreter, op: irdl.ParametricOp, args: PythonValues
    ):
        base_type = self.attrs[op.base_type.root_reference]
        constr = ParamAttrConstraint(base_type, args)
        if len(op.output.uses) > 1:
            constr = self.variable_wrap(constr)
        return (constr,)

    @impl(irdl.TypeOp)
    def run_type(self, interpreter: Interpreter, op: irdl.TypeOp, args: PythonValues):
        dialect_op = cast(irdl.DialectOp, op.parent_op())
        dialect_name = dialect_op.sym_name.data
        self.current = op.sym_name
        self.attrs_defs[self.current] = ParamAttrDef(
            f"{dialect_name}.{op.sym_name.data}", []
        )
        interpreter.run_ssacfg_region(op.body, ())
        for k, v in get_accessors_from_param_attr_def(
            self.attrs_defs[self.current]
        ).items():
            setattr(self.attrs[op.sym_name], k, v)
        setattr(self.attrs[op.sym_name], "name", f"{dialect_name}.{op.sym_name.data}")
        return ()

    @impl(irdl.OperandsOp)
    def run_operands(
        self, interpreter: Interpreter, op: irdl.OperandsOp, args: PythonValues
    ):
        self.ops_defs[self.current].operands = list(
            (f"o{i}", a) for i, a in enumerate(args)
        )
        return ()

    @impl(irdl.ResultsOp)
    def run_results(
        self, interpreter: Interpreter, op: irdl.ResultsOp, args: PythonValues
    ):
        self.ops_defs[self.current].results = list(
            (f"r{i}", a) for i, a in enumerate(args)
        )
        return ()

    @impl(irdl.OperationOp)
    def run_operation(
        self, interpreter: Interpreter, op: irdl.OperationOp, args: PythonValues
    ):
        dialect_op = cast(irdl.DialectOp, op.parent_op())
        dialect_name = dialect_op.sym_name.data
        self.current = op.sym_name
        self.ops_defs[self.current] = OpDef(f"{dialect_name}.{op.sym_name.data}")
        interpreter.run_ssacfg_region(op.body, ())
        for k, v in get_accessors_from_op_def(
            self.ops_defs[self.current], None
        ).items():
            setattr(self.ops[op.sym_name], k, v)
        setattr(self.ops[op.sym_name], "name", f"{dialect_name}.{op.sym_name.data}")
        return ()

    @impl(irdl.ParametersOp)
    def run_parameters(
        self, interpreter: Interpreter, op: irdl.ParametersOp, args: PythonValues
    ):
        self.attrs_defs[self.current].parameters = list(
            (f"p{i}", a) for i, a in enumerate(args)
        )
        return ()

    @impl(irdl.DialectOp)
    def run_dialect(
        self, interpreter: Interpreter, op: irdl.DialectOp, args: PythonValues
    ):
        self.ops: dict[StringAttr, type[IRDLOperation]] = {}
        self.attrs: dict[StringAttr, type[ParametrizedAttribute]] = {}
        self.ops_defs: dict[StringAttr, OpDef] = {}
        self.attrs_defs: dict[StringAttr, ParamAttrDef] = {}

        for entry in op.body.block.ops:
            match entry:
                case irdl.OperationOp():
                    self.ops[entry.sym_name] = type.__new__(
                        type(IRDLOperation),
                        entry.sym_name.data,
                        IRDLOperation.__mro__,
                        dict(IRDLOperation.__dict__),
                    )

                case irdl.TypeOp():
                    self.attrs[entry.sym_name] = type.__new__(
                        type(ParametrizedAttribute),
                        entry.sym_name.data,
                        ParametrizedAttribute.__mro__,
                        dict(ParametrizedAttribute.__dict__),
                    )

                case _:
                    pass
        interpreter.run_ssacfg_region(op.body, ())
        self.dialects(interpreter)[op.sym_name.data] = Dialect(
            op.sym_name.data, list(self.ops.values()), list(self.attrs.values())
        )
        return ()


def make_dialect(op: irdl.DialectOp) -> Dialect:
    module = op.get_toplevel_object()
    if not isinstance(module, ModuleOp):
        raise ValueError("Expected dialect to be nested in a ModuleOp")

    interpreter = Interpreter(module)
    irdl_impl = IRDLFunctions()
    interpreter.register_implementations(irdl_impl)
    interpreter.run_op(op, ())
    return interpreter.get_data(IRDLFunctions, "irdl.dialects", dict)[op.sym_name.data]
