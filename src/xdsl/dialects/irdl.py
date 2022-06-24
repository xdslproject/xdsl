
from __future__ import annotations
from ast import arguments
from dataclasses import dataclass
from lib2to3.pgen2.token import OP
# from tests.operation_builder_test import VarResultOp
from xml.dom.minicompat import StringTypes
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from xdsl.irdl import *
from xdsl.ir import *
from typing import overload

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
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
    name = "equality_type_constraint"
    data: ParameterDef[Attribute]

    def enforce_one_param(data: Attribute) -> EqTypeConstraintAttr:
        if len(data) == 1:
            return data
        else:
            raise Exception(f"Expected 1 parameter but got {len(data)}")

@irdl_attr_definition
class AnyTypeConstraintAttr(ParametrizedAttribute):
    name = "any_type_constraint"
    data: ParameterDef[Attribute]


@irdl_attr_definition
class AnyOfTypeConstraintAttr(ParametrizedAttribute):
    name = "any_of_type"
    first: ParameterDef[Attribute]
    second: ParameterDef[Attribute]

@irdl_attr_definition
class VarTypeConstraintAttr(ParametrizedAttribute):
    name = "var_type_constraint"
    data: ParameterDef[Attribute]
    

@irdl_attr_definition
class DynTypeBaseConstraintAttr(ParametrizedAttribute):
    name = "dyn_type_constraint"
    data: ParameterDef[StringAttr]

@irdl_attr_definition
class DynTypeParamsConstraintAttr(ParametrizedAttribute):
    name = "dyn_type_params_constraint"
    data: ParameterDef[StringAttr]
    params: ParameterDef[ArrayAttr]

@irdl_attr_definition
class TypeParamsConstraintAttr(ParametrizedAttribute):
    name = "type_params_constraint"
    data: ParameterDef[Attribute]

@irdl_attr_definition
class NamedTypeConstraintAttr(ParametrizedAttribute):
    name = "named_type_constraint"

    type_name: ParameterDef[StringAttr]
    params_constraints: ParameterDef[Attribute]
    

@irdl_op_definition
class DialectOp(Operation):
    name: str = "irdl.dialect"
    dialect_name = AttributeDef(StringAttr)
    body = SingleBlockRegionDef()

@irdl_op_definition
class ParametersOp(Operation):
    name: str = "irdl.parameters"
    constraints = AttributeDef(ArrayAttr)


@irdl_op_definition
class TypeOp(Operation):
    name: str = "irdl.type"
    type_name = AttributeDef(StringAttr)
    body = SingleBlockRegionDef()

@irdl_op_definition
class ConstraintVarsOp(Operation):
    name: str = "irdl.constraint_vars"
    constraints = AttributeDef(AnyAttr())

@irdl_op_definition
class OperandsOp(Operation):
    name: str = "irdl.operands"
    op = VarOperandDef(AnyAttr())

@irdl_op_definition
class ResultsOp(Operation):
    name: str = "irdl.results"
    res = VarResultDef(AnyAttr())

@irdl_op_definition
class OperationOp(Operation):
    name: str = "irdl.operation"
    body = SingleBlockRegionDef()


