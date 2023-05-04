from typing import IO
from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Operation
from xdsl.irdl import AttrConstraint
from xdsl.printer import Printer
from xdsl.dialects.irdl import (
    ConstraintVarsOp,
    DialectOp,
    ParametersOp,
    TypeOp,
    OperandsOp,
    ResultsOp,
    OperationOp,
    EqTypeConstraintAttr,
    AnyTypeConstraintAttr,
    AnyOfTypeConstraintAttr,
    DynTypeBaseConstraintAttr,
    DynTypeParamsConstraintAttr,
    TypeParamsConstraintAttr,
    NamedTypeConstraintAttr,
)

# TODO: remove from "ignore" list in pyproject.toml
# https://github.com/xdslproject/xdsl/issues/572


@dataclass(frozen=True, eq=False)
class IRDLPrinter:
    stream: IO[str]

    def _print(self, s: str, end: str | None = None):
        print(s, file=self.stream, end=end)

    def print_module(self, module: ModuleOp):
        for op in module.walk():
            self.ensure_op_is_irdl_op(op)
        self._print("module {")
        for di in module.walk():
            if isinstance(di, DialectOp):
                IRDLPrinter.print_dialect_definition(self, di)
        self._print("}")

    def ensure_op_is_irdl_op(self, op: Operation):
        if not isinstance(
            op,
            (
                OperationOp | ResultsOp | OperandsOp,
                ConstraintVarsOp | TypeOp | ParametersOp | DialectOp | ModuleOp,
            ),
        ):
            raise Exception(f"Operation {op.name} is not an operation in IRDL")

    def print_type_definition(self, type: TypeOp):
        self._print(f"    {TypeOp.name} {type.type_name.data} {{")
        for param in type.walk():
            if isinstance(param, ParametersOp):
                self.print_parameters_definition(param)

        self._print("    }")

    def print_attr_constraint(self, f: AttrConstraint | Attribute):
        if isinstance(f, AnyOfTypeConstraintAttr):
            self._print("AnyOf", end="<")
            for i in range(len(f.params.data) - 1):
                self.print_attr_constraint(f.params.data[i])
                self._print(", ", end="")
            self.print_attr_constraint(f.params.data[len(f.params.data) - 1])
            self._print(">", end="")

        elif isinstance(f, EqTypeConstraintAttr):
            p = Printer(self.stream)
            p.print(f.type)

        elif isinstance(f, AnyTypeConstraintAttr):
            self._print("Any", end="")

        elif isinstance(f, DynTypeBaseConstraintAttr):
            self._print(f.type_name.data, end="")

        elif isinstance(f, DynTypeParamsConstraintAttr):
            self._print(f.type_name.data, end="")
            self._print("<", end="")
            for i in range(len(f.params.data)):
                self.print_attr_constraint(f.params.data[i])
            self._print(">", end="")

        elif isinstance(f, TypeParamsConstraintAttr):
            self._print(f"{f.type_name}")

        elif isinstance(f, NamedTypeConstraintAttr):
            self._print(f.type_name.data, end="")
            self._print(":", end=" ")
            self.print_attr_constraint(f.params_constraints)

        else:
            raise Exception(f"Invalid Constraint: {type(f)} ")

    def print_parameters_definition(self, param_op: ParametersOp):
        self._print(f"      {ParametersOp.name}", end="(")

        for param in param_op.params.data:
            self._print(f"{param.type_name.data}: ", end="")
            self.print_attr_constraint(param.params_constraints)
        self._print(")")

    def print_operation_definition(self, operation: OperationOp):
        self._print(f"    {OperationOp.name} {operation.attributes['name'].data} {{")

        # Checking for existence of operands
        operand_list = [
            operand_def
            for operand_def in operation.walk()
            if isinstance(operand_def, OperandsOp)
        ]

        if operand_list:
            self.print_operand_definition(operand_list)

        # Checking for existence of results
        result_list = [
            result_list.append(res)
            for res in operation.walk()
            if isinstance(res, ResultsOp)
        ]

        if result_list:
            self.print_result_definition(result_list)

        self._print("    }}")

    def print_result_definition(self, res_list: list[ResultsOp]):
        self._print(f"      {ResultsOp.name}", end="(")

        for i in range(len(res_list)):
            for result in res_list[i].params.data:
                self.print_attr_constraint(result)
                self._print(", ", end="") if i != len(res_list) - 1 else None
        self._print(")")

    def print_operand_definition(self, op_list: list[OperandsOp]):
        self._print(f"      {OperandsOp.name}", end="(")

        for i in range(len(op_list)):
            for ops in op_list[i].params.data:
                self.print_attr_constraint(ops)
                self._print(", ", end="") if i != len(op_list) - 1 else None
        self._print(")")

    def print_dialect_definition(self, di: DialectOp):
        self._print(f"  {DialectOp.name} {di.dialect_name.data} {{")

        for type in di.walk():
            if isinstance(type, TypeOp):
                self.print_type_definition(type)

        for op in di.walk():
            if isinstance(op, OperationOp):
                self.print_operation_definition(type)

        self._print("  }}")
