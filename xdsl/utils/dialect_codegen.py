"""
Functions to print a Python file with the definition of a dialect, given specifications
of its operations and attributes.
"""

import itertools
import subprocess
from collections.abc import Iterable
from io import StringIO
from typing import Literal

from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyInt,
    GenericAttrConstraint,
    GenericRangeConstraint,
    OpDef,
    ParamAttrDef,
    RangeOf,
    SingleOf,
)


def generate_dynamic_attr_class(
    class_name: str, attr: ParamAttrDef, is_type: bool = True
) -> type[Attribute]:
    return type(
        class_name,
        (ParametrizedAttribute,) + ((TypeAttribute,) if is_type else ()),
        dict(ParametrizedAttribute.__dict__) | {"name": attr.name},
    )


def get_constraint_from_range(
    constr: GenericRangeConstraint[Attribute],
    operand_or_result: Literal["operand_def", "result_def"],
) -> tuple[str, GenericAttrConstraint[Attribute]]:
    match constr:
        case SingleOf():
            def_prefix = ""
            inner_constr = constr.constr
        case RangeOf(length=AnyInt()):
            def_prefix = "var_"
            inner_constr = constr.constr
        case _:
            raise Exception(f"Constraint type {constr} not supported")

    return def_prefix + operand_or_result, inner_constr


def typedef_to_class_string(class_name: str, typedef: ParamAttrDef) -> str:
    if typedef.parameters:
        raise NotImplementedError("Attribute parameters not yet implemented")

    return f"""
@irdl_attr_definition
class {class_name}(ParametrizedAttribute, TypeAttribute):
\tname = "{typedef.name}"
    """


def attrdef_to_class_string(class_name: str, attr: ParamAttrDef) -> str:
    if attr.parameters:
        raise NotImplementedError("Attribute parameters not yet implemented")
    return f"""
@irdl_attr_definition
class {class_name}(ParametrizedAttribute):
\tname = "{attr.name}"
    """


def opdef_to_class_string(class_name: str, op: OpDef) -> str:
    if op.accessor_names:
        raise NotImplementedError("Operation accessor_names not yet implemented")

    fields_description = ""

    fields_description += (
        "\n\t".join(
            [
                "{} = {}({})".format(
                    name, *get_constraint_from_range(oper.constr, "operand_def")
                )
                for name, oper in op.operands
            ]
        )
        + "\n\t"
    )

    fields_description += (
        "\n\t".join(
            [
                "{} = {}({})".format(
                    name, *get_constraint_from_range(oper.constr, "result_def")
                )
                for name, oper in op.results
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
