from types import NoneType

from xdsl.dialects import stencil
from xdsl.dialects.builtin import ArrayOfConstraint
from xdsl.ir import Dialect, ParametrizedAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    IRDLOperation,
    OperandDef,
    OptOperandDef,
    RangeConstraint,
    SingleOf,
    VarOperandDef,
)


# TODO: dataclass the hell out of this
def constraint_stub(constraint: AttrConstraint):
    match constraint:
        case BaseAttr(attr_type):
            return attr_type.__name__
        case AnyOf(constraints):
            return f"AnyOf({', '.join(constraint_stub(c) for c in constraints)})"
        case AllOf(constraints):
            return f"AllOf({', '.join(constraint_stub(c) for c in constraints)})"
        case ArrayOfConstraint(constraint):
            return f"ArrayOfConstraint({constraint_stub(constraint)})"
        case AnyAttr():
            return "AnyAttr()"
        case _:
            raise NotImplementedError(
                f"Unsupported constraint type: {type(constraint)}"
            )


def range_constraint_stub(constraint: RangeConstraint):
    match constraint:
        case SingleOf(constr):
            return f"SingleOf({constraint_stub(constr)})"
        case _:
            raise NotImplementedError(
                f"Unsupported constraint type: {type(constraint)}"
            )


def attribute_stub(attr: type[ParametrizedAttribute]):
    yield f"class {attr.__name__}(ParametrizedAttribute):"
    attr_def = attr.get_irdl_definition()
    if isinstance(attr_def, NoneType):
        yield "    pass"
    for name, param in attr_def.parameters:
        yield f"    {name} = attr_def({constraint_stub(param)})"
    yield ""
    yield ""


def operation_stub(op: type[IRDLOperation]):
    had_body = False
    yield f"class {op.__name__}(IRDLOperation):"
    op_def = op.get_irdl_definition()
    for name, o in op_def.operands:
        match o:
            case VarOperandDef(constr):
                yield f"    {name} = var_operand_def({range_constraint_stub(constr)})"
            case OptOperandDef(constr):
                yield f"    {name} = opt_operand_def({range_constraint_stub(constr)})"
            case OperandDef(constr):
                yield f"    {name} = operand_def({range_constraint_stub(constr)})"

    if not had_body:
        yield "    pass"
    yield ""
    yield ""


def dialect_stubs(dialect: Dialect):
    for attr in dialect.attributes:
        if issubclass(attr, ParametrizedAttribute):
            yield from attribute_stub(attr)
    for op in dialect.operations:
        if issubclass(op, IRDLOperation):
            yield from operation_stub(op)


if __name__ == "__main__":
    for l in dialect_stubs(stencil.Stencil):
        print(l)
