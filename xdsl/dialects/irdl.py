from __future__ import annotations
from dataclasses import dataclass
from typing import cast, Annotated

from xdsl.dialects.builtin import AnyArrayAttr, ArrayAttr, StringAttr
from xdsl.ir import ParametrizedAttribute, Operation, Attribute, Dialect, OpResult
from xdsl.irdl import (ParameterDef, VarOperand, AnyAttr, AttributeDef,
                       SingleBlockRegionDef, VarResultDef, irdl_op_definition,
                       irdl_attr_definition)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class EqTypeConstraintAttr(ParametrizedAttribute):

    name = "irdl.equality_type_constraint"
    type: ParameterDef[Attribute]


@irdl_attr_definition
class AnyTypeConstraintAttr(ParametrizedAttribute):
    name = "irdl.any_type_constraint"


@irdl_attr_definition
class AnyOfTypeConstraintAttr(ParametrizedAttribute):
    name = "irdl.any_of_type"
    params: ParameterDef[ArrayAttr[Attribute]]


@irdl_attr_definition
class VarTypeConstraintAttr(ParametrizedAttribute):
    name = "irdl.var_type_constraint"
    var_name: ParameterDef[StringAttr]


@irdl_attr_definition
class DynTypeBaseConstraintAttr(ParametrizedAttribute):
    name = "irdl.dyn_type_constraint"
    type_name: ParameterDef[StringAttr]


@irdl_attr_definition
class DynTypeParamsConstraintAttr(ParametrizedAttribute):
    name = "irdl.dyn_type_params_constraint"
    type_name: ParameterDef[StringAttr]
    params: ParameterDef[AnyArrayAttr]


@irdl_attr_definition
class TypeParamsConstraintAttr(ParametrizedAttribute):
    name = "irdl.type_params_constraint"
    type_name: ParameterDef[StringAttr]
    params: ParameterDef[AnyArrayAttr]


@irdl_attr_definition
class NamedTypeConstraintAttr(ParametrizedAttribute):
    name = "irdl.named_type_constraint"
    type_name: ParameterDef[StringAttr]
    params_constraints: ParameterDef[Attribute]

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_char("<")
        type_name = parser.parse_str_literal()
        parser.parse_char(":")
        params_constraints = parser.parse_attribute()
        parser.parse_char(">")
        return [StringAttr.from_str(type_name), params_constraints]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<\"", self.type_name.data, "\" : ",
                      self.params_constraints, ">")


@irdl_op_definition
class DialectOp(Operation):
    """
    Define a new dialect
    """
    name = "irdl.dialect"
    body = SingleBlockRegionDef()

    @property
    def dialect_name(self) -> StringAttr:
        return cast(StringAttr, self.attributes["name"])

    def verify_(self) -> None:
        if "name" not in self.attributes.keys():
            raise ValueError("name attribute is required")
        if not isinstance(self.attributes["name"], StringAttr):
            raise ValueError("name attribute must be a string attribute")

    def get_op_defs(self) -> list[OperationOp]:
        """Get the operations defined by the dialect"""
        return [op for op in self.body.ops if isinstance(op, OperationOp)]

    def get_type_defs(self) -> list[TypeOp]:
        """Get the types defined by the dialect"""
        return [op for op in self.body.ops if isinstance(op, TypeOp)]


@irdl_op_definition
class ParametersOp(Operation):
    """
    Define the parameters of a type/attribute definition
    """
    name = "irdl.parameters"
    params = AttributeDef(ArrayAttr)


@irdl_op_definition
class TypeOp(Operation):
    """
    Defines new types belonging to previously defined dialect
    """
    name = "irdl.type"
    body = SingleBlockRegionDef()

    @property
    def type_name(self) -> StringAttr:
        return cast(StringAttr, self.attributes["name"])

    def verify_(self) -> None:
        if "name" not in self.attributes.keys():
            raise ValueError("name attribute is required")
        if not isinstance(self.attributes["name"], StringAttr):
            raise ValueError("name attribute must be a string attribute")


@irdl_op_definition
class ConstraintVarsOp(Operation):
    """
    Define constraint variables that can be used in the
    current region
    """
    name = "irdl.constraint_vars"
    constraints = AttributeDef(AnyAttr())


@irdl_op_definition
class OperandsOp(Operation):
    """
    Define the operands of a parent operation
    """
    name = "irdl.operands"
    params = AttributeDef(AnyAttr())


@irdl_op_definition
class ResultsOp(Operation):
    """
    Define results of parent operation
    """
    name = "irdl.results"
    params = AttributeDef(AnyAttr())


@irdl_op_definition
class OperationOp(Operation):
    """
    Define a new operation belonging to previously defined dialect
    """
    name = "irdl.operation"
    body = SingleBlockRegionDef()

    @property
    def op_name(self) -> StringAttr:
        return cast(StringAttr, self.attributes["name"])

    def verify_(self) -> None:
        if "name" not in self.attributes.keys():
            raise ValueError("name attribute is required")
        if not isinstance(self.attributes["name"], StringAttr):
            raise ValueError("name attribute must be a string attribute")

    def get_operands(self) -> OperandsOp | None:
        """Get the operation operands definition"""
        for op in self.body.ops:
            if isinstance(op, OperandsOp):
                return op
        return None

    def get_results(self) -> ResultsOp | None:
        """Get the operation results definition"""
        for op in self.body.ops:
            if isinstance(op, ResultsOp):
                return op
        return None


IRDL = Dialect(
    [
        DialectOp,
        ParametersOp,
        TypeOp,
        ConstraintVarsOp,
        OperandsOp,
        ResultsOp,
        OperationOp,
    ],
    [
        AnyTypeConstraintAttr,
        AnyOfTypeConstraintAttr,  #
        EqTypeConstraintAttr,
        VarTypeConstraintAttr,
        TypeParamsConstraintAttr,
        NamedTypeConstraintAttr,
        DynTypeBaseConstraintAttr,
        DynTypeParamsConstraintAttr
    ])
