from dataclasses import dataclass, field
from typing import Annotated, Any

from xdsl.dialects.builtin import ArrayAttr, ArrayOfConstraint
from xdsl.interpreter import Successor
from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
    OptSuccessor,
    ParametrizedAttribute,
    Region,
    VarSuccessor,
)
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    AttributeDef,
    BaseAttr,
    EqAttrConstraint,
    IRDLOperation,
    Operand,
    OperandDef,
    OptAttributeDef,
    OptOperand,
    OptOperandDef,
    OptOpResult,
    OptPropertyDef,
    OptRegion,
    OptRegionDef,
    OptResultDef,
    OptSuccessorDef,
    ParamAttrConstraint,
    PropertyDef,
    RangeConstraint,
    RegionDef,
    ResultDef,
    SingleOf,
    SuccessorDef,
    VarOperand,
    VarOperandDef,
    VarOpResult,
    VarRegion,
    VarRegionDef,
    VarResultDef,
    VarSuccessorDef,
)


@dataclass
class DialectStub:
    dialect: Dialect
    dependencies: dict[Any, set[tuple[Any, str]]] = field(
        init=False, default_factory=dict
    )

    def _import(self, object: Any, alias: str = ""):
        module = object.__module__
        if module == "builtins":
            return
        if module.startswith("xdsl.ir."):
            module = "xdsl.ir"
        if module.startswith("xdsl.irdl."):
            module = "xdsl.irdl"
        if module in self.dependencies:
            self.dependencies[module].add((object, alias))
        else:
            self.dependencies[module] = {(object, alias)}

    def _constraint_type(self, constraint: AttrConstraint) -> str:
        match constraint:
            case BaseAttr(attr_type):
                if attr_type not in self.dialect.attributes:
                    self._import(attr_type)
                return attr_type.__name__
            case EqAttrConstraint(attr):
                if type(attr) not in self.dialect.attributes:
                    self._import(type(attr))
                return type(attr).__name__

            case AnyOf(constraints):
                return " | ".join(self._constraint_type(c) for c in constraints)
            case AllOf(constraints):
                self._import(Annotated)
                return f"Annotated[{', '.join(self._constraint_type(c) for c in reversed(constraints))}]"
            case ArrayOfConstraint(constraint):
                self._import(ArrayAttr)
                return f"ArrayAttr[{self._constraint_type(constraint)}]"
            case AnyAttr():
                self._import(Attribute)
                return "Attribute"
            case ParamAttrConstraint(base_type):
                return base_type.__name__

            case _:
                raise NotImplementedError(
                    f"Unsupported constraint type: {type(constraint)}"
                )

    def _range_constraint_stub(self, constraint: RangeConstraint):
        match constraint:
            case SingleOf(constr):
                return f"SingleOf({self._constraint_type(constr)})"
            case _:
                raise NotImplementedError(
                    f"Unsupported constraint type: {type(constraint)}"
                )

    def _attribute_stub(self, attr: type[ParametrizedAttribute]):
        self._import(ParametrizedAttribute)
        bases = (b for b in set(attr.__mro__[1:]) - set(ParametrizedAttribute.__mro__))
        bases = ", ".join(b.__name__ for b in bases)
        if bases:
            bases += ", "
        yield f"class {attr.__name__}({bases}ParametrizedAttribute):"
        attr_def = attr.get_irdl_definition()
        for name, param in attr_def.parameters:
            yield f'    {name} : "{self._constraint_type(param)}"'
        yield ""
        yield ""

    def _operation_stub(self, op: type[IRDLOperation]):
        had_body = False
        self._import(IRDLOperation)
        yield f"class {op.__name__}(IRDLOperation):"
        op_def = op.get_irdl_definition()
        for name, o in op_def.operands:
            had_body = True
            match o:
                case VarOperandDef(_):
                    self._import(VarOperand, "VarOperand")
                    yield f"    {name} : VarOperand"
                case OptOperandDef(_):
                    self._import(OptOperand, "OptOperand")
                    yield f"    {name} : OptOperand"
                case OperandDef(_):
                    self._import(Operand, "Operand")
                    yield f"    {name} : Operand"
        for name, o in op_def.results:
            had_body = True
            match o:
                case VarResultDef():
                    self._import(VarOpResult)
                    yield f"    {name} : VarOpResult"
                case OptResultDef():
                    self._import(OptOpResult, "OptOpResult")
                    yield f"    {name} : OptOpResult"
                case ResultDef():
                    self._import(OpResult)
                    yield f"    {name} : OpResult"
        for name, o in op_def.attributes.items():
            had_body = True
            match o:
                case OptAttributeDef():
                    yield f"    {name} : {self._constraint_type(o.constr)} | None"
                case AttributeDef():
                    yield f"    {name} : {self._constraint_type(o.constr)}"
        for name, o in op_def.properties.items():
            had_body = True
            match o:
                case OptPropertyDef():
                    yield f"    {name} : {self._constraint_type(o.constr)} | None"
                case PropertyDef():
                    yield f"    {name} : {self._constraint_type(o.constr)}"

        for name, r in op_def.regions:
            had_body = True
            match r:
                case OptRegionDef():
                    self._import(OptRegion, "OptRegion")
                    yield f"    {name} : OptRegion"
                case VarRegionDef():
                    self._import(VarRegion, "VarRegion")
                    yield f"    {name} : VarRegion"
                case RegionDef():
                    self._import(Region)
                    yield f"    {name} : Region"

        for name, r in op_def.successors:
            had_body = True
            match r:
                case OptSuccessorDef():
                    self._import(OptSuccessor, "OptSuccessor")
                    yield f"    {name} : OptSuccessor"
                case VarSuccessorDef():
                    self._import(VarSuccessor, "VarSuccessor")
                    yield f"    {name} : VarSuccessor"
                case SuccessorDef():
                    self._import(Successor)
                    yield f"    {name} : Successor"

        if not had_body:
            yield "    pass"
        yield ""
        yield ""

    def _dialect_stubs(self):
        for attr in self.dialect.attributes:
            if issubclass(attr, ParametrizedAttribute):
                for l in self._attribute_stub(attr):
                    yield l

        for op in self.dialect.operations:
            if issubclass(op, IRDLOperation):
                for l in self._operation_stub(op):
                    yield l

    def _imports(self):
        items = list(self.dependencies.items())
        items.sort()
        for module, objects in items:
            if len(objects) == 1:
                object = objects.pop()
                yield f"from {module} import {object[1] or object[0].__name__}"
            else:
                names = list(o[1] or o[0].__name__ for o in objects)
                names.sort()
                yield f"from {module} import ("
                for o in names:
                    yield f"    {o},"
                yield ")"

    def dialect_stubs(self):
        self._import(Dialect)
        dialect_body = "\n".join(self._dialect_stubs())
        imports = "\n".join(self._imports())
        if imports:
            imports += "\n"

        return f"""\
{imports}
{dialect_body}
{self.dialect.name.capitalize()} : Dialect
"""
