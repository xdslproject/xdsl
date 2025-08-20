import os
import typing
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Any, cast

import xdsl.ir
import xdsl.irdl
from xdsl.ir import Attribute, Dialect, OpResult, ParametrizedAttribute, Region
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    AttributeDef,
    BaseAttr,
    EqAttrConstraint,
    IRDLOperation,
    OperandDef,
    OptAttributeDef,
    OptOperandDef,
    OptPropertyDef,
    OptRegionDef,
    OptResultDef,
    OptSuccessorDef,
    ParamAttrConstraint,
    PropertyDef,
    RangeOf,
    RegionDef,
    ResultDef,
    SuccessorDef,
    VarOperandDef,
    VarRegionDef,
    VarResultDef,
    VarSuccessorDef,
)


@dataclass
class DialectStubGenerator:
    """Generate a typing stub file (.pyi) for a dialect."""

    dialect: Dialect
    dependencies: dict[str, set[str]] = field(
        init=False, default_factory=dict[str, set[str]]
    )

    def _import(self, module: ModuleType | str, name: str | type[Any]):
        """
        Internal helper to keep track of dependencies, to later generate clean import
        statements.
        """
        # If passed a type, use its name.
        if isinstance(name, type):
            name = name.__name__
        # If passed a module, use its name.
        if isinstance(module, ModuleType):
            module = module.__name__
        # Do not import from builtins, those are the implicitely available ones.
        if module == "builtins":
            return
        # Do no import from banned nested modules
        if module.startswith("xdsl.ir."):
            module = "xdsl.ir"
        if module.startswith("xdsl.irdl."):
            module = "xdsl.irdl"
        # Create a module dependency or add a new name to the ones from an existing
        # dependency
        if module in self.dependencies:
            self.dependencies[module].add(name)
        else:
            self.dependencies[module] = {name}

    def _generate_constraint_type(self, constraint: AttrConstraint) -> str:
        """
        Return a type hint for the member constrained by a constraint, by it an
        attribute parameter, or an operation attribute/property.
        """
        import xdsl.dialects.builtin
        import xdsl.ir
        from xdsl.dialects.builtin import ArrayAttr, ArrayOfConstraint

        match constraint:
            case BaseAttr(attr_type):
                if attr_type not in self.dialect.attributes:
                    self._import(attr_type.__module__, attr_type.__name__)
                return attr_type.__name__
            case EqAttrConstraint(attr):
                if type(attr) not in self.dialect.attributes:
                    self._import(type(attr).__module__, type(attr).__name__)
                return type(attr).__name__

            case AnyOf(attr_constrs=constraints):
                return " | ".join(
                    self._generate_constraint_type(c) for c in constraints
                )
            case AllOf(constraints):
                self._import(typing, "Annotated")
                return f"Annotated[{', '.join(self._generate_constraint_type(c) for c in reversed(constraints))}]"  # noqa: E501
            case ArrayOfConstraint(RangeOf(constraint)):
                self._import(xdsl.dialects.builtin, ArrayAttr)
                return f"ArrayAttr[{self._generate_constraint_type(constraint)}]"
            case AnyAttr():
                self._import(xdsl.ir, Attribute)
                return "Attribute"
            case ParamAttrConstraint():
                base_type = cast(
                    ParamAttrConstraint[ParametrizedAttribute], constraint
                ).base_attr
                return base_type.__name__

            case _:
                raise NotImplementedError(
                    f"Unsupported constraint type: {type(constraint)}"
                )

    def _generate_attribute_stub(self, attr: type[ParametrizedAttribute]):
        """
        Generate type stub for an irdl attribute.
        """
        # They all are ParametrizedAttributes.
        self._import(xdsl.ir, ParametrizedAttribute)
        # Get the bases that are not already bases of ParametrizedAttribute.
        bases = set(attr.__mro__[1:]) - set(ParametrizedAttribute.__mro__)
        # Add them as stub dependencies
        for base in bases:
            self._import(base.__module__, base)

        # Also add them to the Attribute class' bases.
        bases = ", ".join(b.__name__ for b in bases)
        if bases:
            bases += ", "
        yield f"class {attr.__name__}({bases}ParametrizedAttribute):"

        # Generate the parameters' stubs, if any
        attr_def = attr.get_irdl_definition()
        for name, param in attr_def.parameters:
            yield f'    {name} : "{self._generate_constraint_type(param.constr)}"'
        # Otherwise, generate a pass for Python's indentation
        if not attr_def.parameters:
            yield "    pass"
        yield ""
        yield ""

    def _generate_operation_stub(self, op: type[IRDLOperation]):
        """
        Generate type stub for an irdl operation.
        """
        # Keep track of whether the operation has any body, to generate a pass if it
        # does not.
        had_body = False

        # They all are IRDLOperations.
        self._import(xdsl.irdl, IRDLOperation)

        # Currently there's nothing that should be a base class or the IRDLOperations.
        # Traits are not supported, and implemented as fields in PyRDL.
        yield f"class {op.__name__}(IRDLOperation):"

        # Generate the constructs' stubs, if any
        op_def = op.get_irdl_definition()
        for name, o in op_def.operands:
            had_body = True
            match o:
                case VarOperandDef(_):
                    self._import(xdsl.irdl, "VarOperand")
                    yield f"    {name} : VarOperand"
                case OptOperandDef(_):
                    self._import(xdsl.irdl, "OptOperand")
                    yield f"    {name} : OptOperand"
                case OperandDef(_):
                    self._import(xdsl.irdl, "Operand")
                    yield f"    {name} : Operand"
        for name, o in op_def.results:
            had_body = True
            match o:
                case VarResultDef():
                    self._import(xdsl.irdl, "VarOpResult")
                    yield f"    {name} : VarOpResult"
                case OptResultDef():
                    self._import(xdsl.irdl, "OptOpResult")
                    yield f"    {name} : OptOpResult"
                case ResultDef():
                    self._import(xdsl.ir, OpResult)
                    yield f"    {name} : OpResult"
        for name, o in op_def.attributes.items():
            had_body = True
            match o:
                case OptAttributeDef():
                    yield f"    {name} : {self._generate_constraint_type(o.constr)} | None"  # noqa: E501
                case AttributeDef():
                    yield f"    {name} : {self._generate_constraint_type(o.constr)}"
        for name, o in op_def.properties.items():
            had_body = True
            match o:
                case OptPropertyDef():
                    yield f"    {name} : {self._generate_constraint_type(o.constr)} | None"  # noqa: E501
                case PropertyDef():
                    yield f"    {name} : {self._generate_constraint_type(o.constr)}"

        for name, r in op_def.regions:
            had_body = True
            match r:
                case OptRegionDef():
                    self._import(xdsl.irdl, "OptRegion")
                    yield f"    {name} : OptRegion"
                case VarRegionDef():
                    self._import(xdsl.irdl, "VarRegion")
                    yield f"    {name} : VarRegion"
                case RegionDef():
                    self._import(xdsl.ir, Region)
                    yield f"    {name} : Region"

        for name, r in op_def.successors:
            had_body = True
            match r:
                case OptSuccessorDef():
                    self._import(xdsl.irdl, "OptSuccessor")
                    yield f"    {name} : OptSuccessor"
                case VarSuccessorDef():
                    self._import(xdsl.irdl, "VarSuccessor")
                    yield f"    {name} : VarSuccessor"
                case SuccessorDef():
                    self._import(xdsl.irdl, "Successor")
                    yield f"    {name} : Successor"
        # Generate a pass if the operation had no body.
        if not had_body:
            yield "    pass"
        yield ""
        yield ""

    def _generate_dialect_stubs(self):
        """
        Generate a dialect's stubs.

        Just generate stubs for all attributes and operations in the dialect.
        """
        for attr in self.dialect.attributes:
            if issubclass(attr, ParametrizedAttribute):
                for l in self._generate_attribute_stub(attr):
                    yield l

        for op in self.dialect.operations:
            if issubclass(op, IRDLOperation):
                for l in self._generate_operation_stub(op):
                    yield l

    def _generate_imports(self):
        """
        Generate import statements for all the dependencies of the stub.
        """
        # sort modules alphabetically for deterministic and clean output.
        items = list(self.dependencies.items())
        items.sort()

        for module, names in items:
            # If only one name is imported from a module, make a one-liner import.
            if len(names) == 1:
                name = names.pop()
                yield f"from {module} import {name}"
            # Otherwise, import all names in a multi-line import, sorted again for
            # a deterministic and clean output.
            else:
                names = list(names)
                names.sort()
                yield f"from {module} import ("
                for o in names:
                    yield f"    {o},"
                yield ")"

    def generate_dialect_stubs(self):
        """
        The main function, generate stubs for the passed dialect and return as a string.

        NB: probably not optimal perf-wise, but I don't foresee this as a bottleneck.
        """
        self._import(xdsl.ir, Dialect)
        dialect_body = "\n".join(self._generate_dialect_stubs())
        imports = "\n".join(self._generate_imports())
        if imports:
            imports += "\n"

        return f"""\
{imports}
{dialect_body}
{self.dialect.name.capitalize()} : Dialect
"""


def make_all_stubs():
    import xdsl.dialects

    dialects = xdsl.dialects
    directory = "/".join(dialects.__path__)
    for file in os.listdir(directory):
        name, ext = os.path.splitext(file)
        if ext == ".irdl":
            import_module(f"{directory}/{name}")


if __name__ == "__main__":
    make_all_stubs()
