"""
Functions to print a Python file with the definition of a dialect, given specifications
of its operations and attributes.
"""

import itertools
import subprocess
from collections.abc import Iterable
from io import StringIO

from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyInt,
    OpDef,
    OperandDef,
    OptOperandDef,
    OptResultDef,
    ParamAttrDef,
    RangeOf,
    ResultDef,
    SingleOf,
    VarOperandDef,
    VarResultDef,
)


def generate_dynamic_attr_class(
    class_name: str, attr: ParamAttrDef, is_type: bool = True
) -> type[Attribute]:
    """
    Dynamically define a type based on ParamAttrDef.
    This is needed to reference dynamically created attributes in operations.
    """
    return type(
        class_name,
        (ParametrizedAttribute,) + ((TypeAttribute,) if is_type else ()),
        dict(ParametrizedAttribute.__dict__) | {"name": attr.name},
    )


def get_str_from_operand_or_result(
    name: str, operand_or_result: OperandDef | ResultDef
) -> str:
    """
    Get a constraint from the GenericRangeConstraint wrapper.
    Build the correct definition function based on the wrapper's type.
    """
    match operand_or_result.constr:
        case SingleOf():
            inner_constr = operand_or_result.constr.constr
        case RangeOf(length=AnyInt()):
            inner_constr = operand_or_result.constr.constr
        case _:
            raise NotImplementedError(
                f"Constraint type {operand_or_result.constr} not yet implemented"
            )

    match operand_or_result:
        case VarOperandDef():
            def_str = "var_operand_def"
        case OptOperandDef():
            def_str = "opt_operand_def"
        case OperandDef():
            def_str = "operand_def"
        case VarResultDef():
            def_str = "var_result_def"
        case OptResultDef():
            def_str = "opt_result_def"
        case ResultDef():
            def_str = "result_def"

    return f"{name} = {def_str}({inner_constr})"


def typedef_to_class_string(class_name: str, typedef: ParamAttrDef) -> str:
    """
    Generate class definition for a type.
    """
    if typedef.parameters:
        raise NotImplementedError("Attribute parameters not yet implemented")

    return f"""
@irdl_attr_definition
class {class_name}(ParametrizedAttribute, TypeAttribute):
\tname = "{typedef.name}"
    """


def attrdef_to_class_string(class_name: str, attr: ParamAttrDef) -> str:
    """
    Generate class definition for an attribute.
    """
    if attr.parameters:
        raise NotImplementedError("Attribute parameters not yet implemented")
    return f"""
@irdl_attr_definition
class {class_name}(ParametrizedAttribute):
\tname = "{attr.name}"
    """


def opdef_to_class_string(class_name: str, op: OpDef) -> str:
    """
    Generate class definition for an operation.
    """
    if op.accessor_names:
        raise NotImplementedError("Operation accessor_names not yet implemented")

    fields_description = ""

    fields_description += (
        "\n\t".join(
            [
                get_str_from_operand_or_result(name, operand_or_result)
                for name, operand_or_result in itertools.chain(op.operands, op.results)
            ]
        )
        + "\n\t"
    )

    if op.attributes:
        raise NotImplementedError("Operation attributes not yet implemented")
    if op.regions:
        raise NotImplementedError("Operation regions not yet implemented")
    if op.successors:
        raise NotImplementedError("Operation successors not yet implemented")
    if op.traits.traits:
        raise NotImplementedError(f"Operation traits not yet implemented {op.traits}")
    if op.properties:
        raise NotImplementedError("Operation properties not yet implemented")

    return f"""
@irdl_op_definition
class {class_name}(IRDLOperation):
\tname = "{op.name}"
\t{fields_description}
\t{f'assembly_format = "{op.assembly_format}"' if op.assembly_format else ""}
    """


def dump_dialect_pyfile(
    dialect_name: str,
    ops: Iterable[tuple[str, OpDef]] = (),
    *,
    attributes: Iterable[tuple[str, ParamAttrDef]] = (),
    types: Iterable[tuple[str, ParamAttrDef]] = (),
    out: StringIO | None = None,
    dialect_obj_name: str = "",
):
    """
    Generate a python file with a dialect comprised of given ops, attributes and types.
    """
    if not dialect_obj_name:
        dialect_obj_name = dialect_name.capitalize() + "Dialect"

    imports = """
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *

# ruff: noqa: F403, F405
    """

    types_class_defs = "\n".join(
        typedef_to_class_string(class_name, attr) for class_name, attr in types
    )

    attr_class_defs = "\n".join(
        attrdef_to_class_string(class_name, attr) for class_name, attr in attributes
    )

    op_class_defs = "\n".join(
        opdef_to_class_string(class_name, op) for class_name, op in ops
    )

    op_list = ",".join(name for name, _ in ops)
    attr_list = ",".join(name for name, _ in itertools.chain(attributes, types))

    dialect_def = (
        f'{dialect_obj_name} = Dialect("{dialect_name}", [{op_list}], [{attr_list}])'
    )

    content = "\n".join(
        (
            imports,
            types_class_defs,
            attr_class_defs,
            op_class_defs,
            dialect_def,
        )
    )
    # Format output
    output = subprocess.run(
        [
            "ruff",
            "format",
            "--stdin-filename",
            f"{dialect_name}.py",
        ],
        input=content,
        capture_output=True,
        text=True,
    )

    print(output.stdout, file=out, end="")
