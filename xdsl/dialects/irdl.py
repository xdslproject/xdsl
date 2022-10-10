from __future__ import annotations
from dataclasses import dataclass
from typing import cast
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from xdsl.irdl import (ParameterDef, VarOperandDef, AnyAttr, AttributeDef,
                       SingleBlockRegionDef, VarResultDef, irdl_op_definition,
                       irdl_attr_definition)
from xdsl.ir import ParametrizedAttribute, Operation, MLContext, Attribute
from xdsl.parser import Parser
from xdsl.printer import Printer


@dataclass
class IRDL:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(EqTypeConstraintAttr)
        self.ctx.register_attr(AnyTypeConstraintAttr)
        self.ctx.register_attr(AnyOfTypeConstraintAttr)
        self.ctx.register_attr(VarTypeConstraintAttr)
        self.ctx.register_attr(DynTypeBaseConstraintAttr)
        self.ctx.register_attr(DynTypeParamsConstraintAttr)
        self.ctx.register_attr(TypeParamsConstraintAttr)
        self.ctx.register_attr(NamedTypeConstraintAttr)

        self.ctx.register_op(DialectOp)
        self.ctx.register_op(ParametersOp)
        self.ctx.register_op(TypeOp)
        self.ctx.register_op(ConstraintVarsOp)
        self.ctx.register_op(OperandsOp)
        self.ctx.register_op(ResultsOp)
        self.ctx.register_op(OperationOp)


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
    params: ParameterDef[ArrayAttr]


@irdl_attr_definition
class TypeParamsConstraintAttr(ParametrizedAttribute):
    name = "irdl.type_params_constraint"
    type_name: ParameterDef[StringAttr]
    params: ParameterDef[ArrayAttr]


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
    op = VarOperandDef(AnyAttr())
    params = AttributeDef(AnyAttr())


@irdl_op_definition
class ResultsOp(Operation):
    """
    Define results of parent operation
    """
    name = "irdl.results"
    res = VarResultDef(AnyAttr())
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