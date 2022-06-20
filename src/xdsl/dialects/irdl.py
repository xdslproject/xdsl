"""
irdl:

EqTypeConstraintAttr and its parameters: type
AnyTypeConstraintAttr and its parameters: -
AnyOfTypeConstraintAttr and its parameters: constrs
VarTypeConstraintAttr and its parameters: name
DynTypeBaseConstraintAttr and its parameters: typeName
DynTypeParamsConstraintAttr and its parameters: typeName, paramConstraints
TypeParamsConstraintAttr and its parameters: typeDef, paramConstraints
NamedTypeConstraintAttr and its parameters: name, constraint

"""

from __future__ import annotations
from ast import arguments
from dataclasses import dataclass
from lib2to3.pgen2.token import OP
# from tests.operation_builder_test import VarResultOp
from xml.dom.minicompat import StringTypes
from xdsl.dialects.builtin import StringAttr

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
    # data: AnyAttr()
    element_type = ParameterDef[AnyAttr()]

    # if this is aproved then I can use it to enforce parameters in other ßclasses 
    def enforce_one_param(data: AnyAttr()) -> EqTypeConstraintAttr:
        if len(data) == 1:
            return data
        else:
            raise Exception(f"Expected 1 parameter but got {len(data)}")
            # ^^ eventually make use of Exception class Shaolun is building

@irdl_attr_definition
class AnyTypeConstraintAttr(ParametrizedAttribute):
    name = "any_type_constraint"
    element_type = ParameterDef[AnyAttr()]


@irdl_attr_definition
class AnyOfTypeConstraintAttr(ParametrizedAttribute):
    name = "any_of_type"
    element_type = ParameterDef[AnyAttr()]

@irdl_attr_definition
class VarTypeConstraintAttr(ParametrizedAttribute):
    name = "var_type_constraint"
    element_type = ParameterDef[AnyAttr()]
    

@irdl_attr_definition
class DynTypeBaseConstraintAttr(ParametrizedAttribute):
    name = "dyn_type_constraint"
    element_type = ParameterDef[StringAttr]

@irdl_attr_definition
class DynTypeParamsConstraintAttr(ParametrizedAttribute):
    name = "dyn_type_params_constraint"
    element_type = ParameterDef[StringAttr]

@irdl_attr_definition
class TypeParamsConstraintAttr(ParametrizedAttribute):
    name = "type_params_constraint"
    element_type = ParameterDef[AnyAttr()]

@irdl_attr_definition
class NamedTypeConstraintAttr(ParametrizedAttribute):
    "named_type_constraint"
    element_type = ParameterDef[AnyAttr()]
    # ^^ seems wrong so this is what I think ... 
# in TableGen "let parameters = (ins StringRefParameter<>:$name, TypeConstraintParameter:$constraint);"
# so would it be: element_type = ParameterDef(TypeParamsConstraintAttr(),StringAttr())


@irdl_op_definition
class DialectOp(Operation):
    name: str = "irdl.dialect"
    arguments = OperandDef(StringAttr)
    body = SingleBlockRegionDef()

@irdl_op_definition
class ParametersOp(Operation):
    name: str = "irdl.parameters"
    arguments = VarOperandDef(AnyAttr())


@irdl_op_definition
class TypeOp(Operation):
    name: str = "irdl.type"
    arguments = OperandDef(StringAttr)
    body = SingleBlockRegionDef()

@irdl_op_definition
class ConstraintVarsOp(Operation):
    name = str = "irdl.constraint_vars"
    arguments = VarOperandDef(AnyAttr())


@irdl_op_definition
class OperandsOp(Operation):
    name = str = "irdl.operands"
    arguments = VarOperandDef(AnyAttr())


@irdl_op_definition
class ResultsOp(Operation):
    name = str = "irdl.results"
    arguments = VarResultDef(AnyAttr())

@irdl_op_definition
class OperationOp(Operation):
    name = str = "irdl.operation"
    arguments = OperandDef(StringAttr)

