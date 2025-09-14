"""
Translate an IRDL program to a Python program creating the corresponding xDSL dialects.
"""

import keyword

from xdsl.dialects.irdl import (
    AttributeOp,
    DialectOp,
    OperandsOp,
    OperationOp,
    ParametersOp,
    ResultsOp,
    TypeOp,
)
from xdsl.dialects.irdl.irdl import VariadicityAttr


def python_name(name: str):
    if keyword.iskeyword(name):
        return f"{name}_"
    return name


def convert_type_or_attr(op: TypeOp | AttributeOp, dialect_name: str) -> str:
    """
    Convert an IRDL type or attribute to Python code creating
    that type or attribute in xDSL.
    """
    type_addition = ", TypeAttribute" if isinstance(op, TypeOp) else ""
    res = f"""\
@irdl_attr_definition
class {op.sym_name.data}(ParametrizedAttribute{type_addition}):
    name = "{dialect_name}.{op.sym_name.data}"
"""

    for sub_op in op.body.ops:
        if not isinstance(sub_op, ParametersOp):
            continue
        for name in sub_op.names:
            res += f"    {python_name(name.data)}: Attribute\n"
    return res


def convert_op(op: OperationOp, dialect_name: str) -> str:
    """Convert an IRDL operation to Python code creating that operation in xDSL."""
    res = f"""\
@irdl_op_definition
class {op.get_py_class_name()}(IRDLOperation):
    name = "{dialect_name}.{op.sym_name.data}"
"""

    for sub_op in op.body.ops:
        if isinstance(sub_op, OperandsOp):
            for name, var in zip(sub_op.names, sub_op.variadicity.value):
                py_name = python_name(name.data)
                match var:
                    case VariadicityAttr.SINGLE:
                        res += f"    {py_name} = operand_def()\n"
                    case VariadicityAttr.OPTIONAL:
                        res += f"    {py_name} = opt_operand_def()\n"
                    case VariadicityAttr.VARIADIC:
                        res += f"    {py_name} = var_operand_def()\n"
                    case _:
                        pass

        if isinstance(sub_op, ResultsOp):
            for name, var in zip(sub_op.names, sub_op.variadicity.value):
                py_name = python_name(name.data)
                match var:
                    case VariadicityAttr.SINGLE:
                        res += f"    {py_name} = result_def()\n"
                    case VariadicityAttr.OPTIONAL:
                        res += f"    {py_name} = opt_result_def()\n"
                    case VariadicityAttr.VARIADIC:
                        res += f"    {py_name} = var_result_def()\n"
                    case _:
                        pass
    res += "    regs = var_region_def()\n"
    res += "    succs = var_successor_def()\n"
    return res


def convert_dialect(dialect: DialectOp) -> str:
    """Convert an IRDL dialect to Python code creating that dialect in xDSL."""
    res = ""
    ops: list[str] = []
    attrs: list[str] = []
    for op in dialect.body.ops:
        if isinstance(op, TypeOp) or isinstance(op, AttributeOp):
            res += convert_type_or_attr(op, dialect.sym_name.data) + "\n\n"
            attrs += [op.sym_name.data]
        if isinstance(op, OperationOp):
            res += convert_op(op, dialect.sym_name.data) + "\n\n"
            ops += [op.get_py_class_name()]
    op_list = "[" + ", ".join(ops) + "]"
    attr_list = "[" + ", ".join(attrs) + "]"
    return (
        res
        + dialect.sym_name.data
        + f' = Dialect("{dialect.sym_name.data}", {op_list}, {attr_list})'
    )
