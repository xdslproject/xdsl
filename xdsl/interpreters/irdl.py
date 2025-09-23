from typing import cast

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.irdl import irdl
from xdsl.dialects.irdl.irdl_to_pyrdl import python_name
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.ir import Attribute, Dialect, Operation, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    AttrConstraint,
    EqAttrConstraint,
    IRDLOperation,
    OpDef,
    OperandDef,
    ParamAttrConstraint,
    ParamAttrDef,
    ParamDef,
    ResultDef,
    VarConstraint,
    get_accessors_from_op_def,
    get_accessors_from_param_attr_def,
)
from xdsl.traits import SymbolTable


@register_impls
class IRDLFunctions(InterpreterFunctions):
    @staticmethod
    def get_dialect(interpreter: Interpreter, name: str) -> Dialect:
        """
        Get a dialect by name from the interpreter's state
        """
        return interpreter.get_data(IRDLFunctions, "irdl.dialects", dict)[name]

    @staticmethod
    def set_dialect(interpreter: Interpreter, name: str, dialect: Dialect) -> None:
        """
        Set or create a dialect by name in the interpreter's state
        """
        interpreter.get_data(IRDLFunctions, "irdl.dialects", dict)[name] = dialect

    @staticmethod
    def get_op(interpreter: Interpreter, name: str) -> type[Operation]:
        """
        Get an operation type by name from the interpreter's state
        """
        ops = IRDLFunctions.get_dialect(
            interpreter, Dialect.split_name(name)[0]
        ).operations
        for op in ops:
            if op.name == name:
                return op
        raise ValueError(f"Operation {name} not found")

    @staticmethod
    def get_attr(interpreter: Interpreter, name: str) -> type[ParametrizedAttribute]:
        """
        Get an attribute type by name from the interpreter's state
        """
        attrs = IRDLFunctions.get_dialect(
            interpreter, Dialect.split_name(name)[0]
        ).attributes
        for attr in attrs:
            if attr.name == name:
                if not issubclass(attr, ParametrizedAttribute):
                    raise ValueError(f"Attribute {name} is not parametrized")
                return attr
        raise ValueError(f"Attribute {name} not found")

    @staticmethod
    def _get_op_def(interpreter: Interpreter, name: str) -> OpDef:
        """
        Get an operation definition by name from the interpreter's state
        """
        return interpreter.get_data(IRDLFunctions, "irdl.op_defs", dict)[name]

    @staticmethod
    def _set_op_def(interpreter: Interpreter, name: str, op_def: OpDef) -> None:
        """
        Set or create an operation definition by name in the interpreter's state
        """
        interpreter.get_data(IRDLFunctions, "irdl.op_defs", dict)[name] = op_def

    @staticmethod
    def _get_attr_def(interpreter: Interpreter, name: str) -> ParamAttrDef:
        """
        Get an attribute definition by name from the interpreter's state
        """
        return interpreter.get_data(IRDLFunctions, "irdl.attr_defs", dict)[name]

    @staticmethod
    def _set_attr_def(
        interpreter: Interpreter, name: str, attr_def: ParamAttrDef
    ) -> None:
        """
        Set or create an attribute definition by name in the interpreter's state
        """
        interpreter.get_data(IRDLFunctions, "irdl.attr_defs", dict)[name] = attr_def

    @staticmethod
    def next_variable_counter(interpreter: Interpreter) -> int:
        counter = interpreter.get_data(
            IRDLFunctions, "irdl.variable_counters", lambda: 0
        )
        interpreter.set_data(IRDLFunctions, "irdl.variable_counters", counter + 1)
        return counter

    @staticmethod
    def variable_wrap(interpreter: Interpreter, constr: AttrConstraint):
        counter = IRDLFunctions.next_variable_counter(interpreter)
        return VarConstraint(f"V{counter}", constr)

    @impl(irdl.IsOp)
    def run_is(self, interpreter: Interpreter, op: irdl.IsOp, args: PythonValues):
        constr = EqAttrConstraint(op.expected)
        if op.output.has_more_than_one_use():
            constr = self.variable_wrap(interpreter, constr)
        return (constr,)

    @impl(irdl.AnyOfOp)
    def run_any_of(
        self, interpreter: Interpreter, op: irdl.AnyOfOp, args: PythonValues
    ):
        constr = AnyOf[Attribute](args)
        if op.output.has_more_than_one_use():
            constr = self.variable_wrap(interpreter, constr)
        return (constr,)

    @impl(irdl.AnyOp)
    def run_any(self, interpreter: Interpreter, op: irdl.AnyOp, args: PythonValues):
        constr = AnyAttr()
        if op.output.has_more_than_one_use():
            constr = self.variable_wrap(interpreter, constr)
        return (constr,)

    @impl(irdl.ParametricOp)
    def run_parametric(
        self, interpreter: Interpreter, op: irdl.ParametricOp, args: PythonValues
    ):
        base_attr_op = SymbolTable.lookup_symbol(op, op.base_type)
        if not isinstance(base_attr_op, irdl.AttributeOp | irdl.TypeOp):
            raise ValueError(
                f"Expected AttributeOp or TypeOp, got {type(base_attr_op)}"
            )
        base_type = self.get_attr(interpreter, base_attr_op.qualified_name)
        constr = ParamAttrConstraint(base_type, args)
        if op.output.has_more_than_one_use():
            constr = self.variable_wrap(interpreter, constr)
        return (constr,)

    @impl(irdl.TypeOp)
    def run_type(self, interpreter: Interpreter, op: irdl.TypeOp, args: PythonValues):
        name = op.qualified_name
        self._set_attr_def(interpreter, name, ParamAttrDef(name, []))
        interpreter.run_ssacfg_region(op.body, ())

        attr = self.get_attr(interpreter, name)
        attr_def = self._get_attr_def(interpreter, name)
        to_inject = get_accessors_from_param_attr_def(attr_def)
        for k, v in to_inject.items():
            setattr(attr, k, v)
        return ()

    @impl(irdl.OperandsOp)
    def run_operands(
        self, interpreter: Interpreter, op: irdl.OperandsOp, args: PythonValues
    ):
        op_op = cast(irdl.OperationOp, op.parent_op())
        op_name = op_op.qualified_name
        self._get_op_def(interpreter, op_name).operands = list(
            (python_name(name.data), OperandDef(a)) for name, a in zip(op.names, args)
        )
        return ()

    @impl(irdl.ResultsOp)
    def run_results(
        self, interpreter: Interpreter, op: irdl.ResultsOp, args: PythonValues
    ):
        op_op = cast(irdl.OperationOp, op.parent_op())
        op_name = op_op.qualified_name
        self._get_op_def(interpreter, op_name).results = list(
            (python_name(name.data), ResultDef(a)) for name, a in zip(op.names, args)
        )
        return ()

    @impl(irdl.OperationOp)
    def run_operation(
        self, interpreter: Interpreter, op: irdl.OperationOp, args: PythonValues
    ):
        name = op.qualified_name
        self._set_op_def(interpreter, name, OpDef(name))
        interpreter.run_ssacfg_region(op.body, ())
        op_def = self._get_op_def(interpreter, name)
        op_type = self.get_op(interpreter, name)

        to_inject = get_accessors_from_op_def(op_def, None)
        for k, v in to_inject.items():
            setattr(op_type, k, v)
        return ()

    @impl(irdl.ParametersOp)
    def run_parameters(
        self, interpreter: Interpreter, op: irdl.ParametersOp, args: PythonValues
    ):
        attr_op = cast(irdl.AttributeOp | irdl.TypeOp, op.parent_op())
        attr_name = attr_op.qualified_name
        self._get_attr_def(interpreter, attr_name).parameters = list(
            (python_name(name.data), ParamDef(a)) for name, a in zip(op.names, args)
        )
        return ()

    @impl(irdl.DialectOp)
    def run_dialect(
        self, interpreter: Interpreter, op: irdl.DialectOp, args: PythonValues
    ):
        operations: list[type[Operation]] = []
        attributes: list[type[Attribute]] = []
        for entry in op.body.block.ops:
            match entry:
                case irdl.OperationOp():
                    operations.append(
                        type(
                            entry.get_py_class_name(),
                            (IRDLOperation,),
                            dict(IRDLOperation.__dict__)
                            | {"name": entry.qualified_name},
                        )
                    )

                case irdl.TypeOp():
                    attributes.append(
                        type(
                            entry.sym_name.data,
                            (TypeAttribute, ParametrizedAttribute),
                            dict(ParametrizedAttribute.__dict__)
                            | {"name": entry.qualified_name},
                        )
                    )

                case _:
                    pass

        self.set_dialect(
            interpreter,
            op.sym_name.data,
            Dialect(op.sym_name.data, operations, attributes),
        )
        interpreter.run_ssacfg_region(op.body, ())
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
