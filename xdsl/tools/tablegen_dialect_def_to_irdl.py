from __future__ import annotations
from dataclasses import dataclass, field
from io import StringIO, TextIOWrapper
from typing import List, Optional, Sequence


@dataclass
class Op():
    name: str
    summary: str = ""
    description: str = ""
    arguments: Sequence[TypedName] = field(
        default_factory=list)  # List of Tuples of name, type
    results: Sequence[TypedName] = field(
        default_factory=list)  # List of Tuples of name, type

    def to_irdl_string(self, dialect_name: str) -> str:
        irdl_string = """"""
        irdl_string += "@irdl_op_definition\n"
        irdl_string += f"class ONNX{self.name}Op(Operation):\n"
        if self.name.startswith(dialect_name):
            adjusted_name = self.name.removeprefix(dialect_name)
        else:
            adjusted_name = self.name
        irdl_string += f"    name: str = \"{dialect_name.lower()}.{adjusted_name}\"\n"
        irdl_string += f"    summary: str = \\\nr\"\"\"{self.summary}\"\"\"\n"
        irdl_string += f"    description: str =\\\nr\"\"\"{self.description}\"\"\"\n"
        for arg in self.arguments:
            def_type: str = "OptAttributeDef" if arg.is_attribute else ("VarOperandDef" if arg.is_variadic else "OperandDef")
            irdl_string += f"    {arg.name} = {def_type}(Attribute) # should actually be {arg.type}\n"
        for result in self.results:
            def_type: str = "ResultDef" if not arg.is_variadic else "VarResultDef"
            irdl_string += f"    {result.name} = {def_type}(Attribute) # should actually be {result.type}\n"

        return irdl_string


@dataclass
class TypedName():
    """
    used for Operands, Results and Attributes
    """
    name: str
    type: str
    is_attribute: bool = False
    is_variadic: bool = False


def get_dialect_def(dialect_name: str,
                    ops: Sequence[Op],
                    include_imports: bool = False) -> str:
    dialect_string = """"""
    dialect_string += "# This file was generated using the script src/tools/tablegen_to_irdl.py. Editing it is a bad idea.\n"
    if include_imports:
        dialect_string += "from __future__ import annotations\n"
        dialect_string += "from xdsl.ir import *\n"
        dialect_string += "from xdsl.irdl import *\n"
        dialect_string += "from xdsl.util import *\n"
        dialect_string += "\n"
        dialect_string += "\n"
    dialect_string += "@dataclass\n"
    dialect_string += f"class {dialect_name}:\n"
    dialect_string += "    ctx: MLContext\n"
    dialect_string += "\n"
    dialect_string += "    def __post_init__(self):\n"
    for op in ops:
        dialect_string += f"        self.ctx.register_op(ONNX{op.name}Op)\n"
    return dialect_string


def open_file(path: str) -> TextIOWrapper:
    # Opening file
    op_def_file = open(path, 'r')
    if op_def_file is not None:
        return op_def_file
    raise Exception("No such file")


def get_next_line(file: TextIOWrapper) -> Optional[str]:
    line = file.readline()
    return line.strip() if line else None


def skip_until(keyword: str, file: TextIOWrapper) -> Optional[str]:
    while True:
        line = get_next_line(file)
        if line is None:
            return None
        if len(
            line.split()) > 1 and line.split()[1] == "MulAddToGemmOptPattern":
            pass
        match line.split():
            case [keyword_, *_] if keyword_ == keyword:
                return line
            case ["let", keyword_, *_] if keyword_ == keyword:
                return line
            case ["def", _, ":", keyword_, *_] if keyword_.startswith(keyword):
                return line
            case _:
                pass
    return None


def parse_text_field(field_name: str, file: TextIOWrapper) -> str:
    line = skip_until(field_name, file)
    if line is None:
        raise Exception(f"did not find text field {field_name}")

    field_contents = line[len("let " + field_name + " = ") + 1:]
    # ; terminates a field
    while line[-1] != ";":
        field_contents += "\n"
        line = get_next_line(file)
        if line is None:
            return field_contents
        field_contents += line

    return field_contents.translate(str.maketrans('', '', '"[{]};'))


def parse_NamedValue_field(field_name: str,
                           file: TextIOWrapper) -> list[TypedName]:
    line = skip_until(field_name, file)
    args: list[TypedName] = []
    if line is None:
        return []
    line = line.lstrip(f"let {field_name} = (ins ")

    def add_ssa_val(cur_line: str):
        type, name = cur_line.split(":")
        is_attr = "Attr" in type
        is_variadic = "Variadic" in type
        args.append(
            TypedName(name.translate(str.maketrans('', '', '$;),')),
                      type,
                      is_attribute=is_attr, is_variadic=is_variadic))

    add_ssa_val(line)
    # ; terminates a field
    while line[-1] != ";":
        line = get_next_line(file)
        if line is None:
            return args
        add_ssa_val(line)

    return args


def parse_op_def(file: TextIOWrapper) -> Optional[Op]:
    """
    skips lines until it finds the beginning of an op_def 
    """
    def_line = skip_until("def", file)
    if def_line is None:
        return None

    # drop def from front, name is before the ":" character
    op = Op(def_line[4:].split("\"")[1])
    op.summary = parse_text_field("summary", file)
    op.description = parse_text_field("description", file)
    op.arguments = parse_NamedValue_field("arguments", file)
    op.results = parse_NamedValue_field("results", file)

    return op


def main():
    file: TextIOWrapper = open_file(
        "/home/martin/development/phd/projects/onnx-mlir/onnx-mlir/src/Dialect/ONNX/ONNXOps.td.inc"
    )
    ops: list[Op] = []
    while op := parse_op_def(file):
        ops.append(op)

    dialect_name = "Onnx"
    file = StringIO()
    print(get_dialect_def(dialect_name, ops, include_imports=True), file=file)
    for op in ops:
        print(op.to_irdl_string(dialect_name=dialect_name), file=file)

    with open(
        "/home/martin/development/phd/projects/xDSL/xdsl/src/xdsl/dialects/onnx.py",
        mode='w') as f:
        print(file.getvalue(), file=f)

    # Just printing the stuff
    # dialect_name = "Onnx"
    # print(get_dialect_def(dialect_name, ops, include_imports=True))
    # for op in ops:
    #     print(op.to_irdl_string(dialect_name=dialect_name))

    # print()


if __name__ == "__main__":
    main()
