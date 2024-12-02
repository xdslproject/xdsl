"""
Translate an IRDL program to a Python program creating the corresponding xDSL dialects.
"""

from xdsl.dialects.irdl import (
    AttributeOp,
    DialectOp,
    OperandsOp,
    OperationOp,
    ParametersOp,
    ResultsOp,
    TypeOp,
)


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
        for idx, _ in enumerate(sub_op.args):
            res += f"    param{idx}: ParameterDef[Attribute]\n"
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
            for idx, _ in enumerate(sub_op.args):
                res += f"    operand{idx} = operand_def()\n"
        if isinstance(sub_op, ResultsOp):
            for idx, _ in enumerate(sub_op.args):
                res += f"    result{idx} = result_def()\n"
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
