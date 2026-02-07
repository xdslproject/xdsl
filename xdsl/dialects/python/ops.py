"""
This file contains the definition of the Python Abstract Syntax Tree (AST) dialect.

The purpose of this dialect is to model Python AST modules.
"""

from collections.abc import Sequence
from pathlib import Path

from xdsl.dialects.builtin import (
    I64,
    FunctionType,
    IntegerAttr,
    StringAttr,
    SymbolNameConstraint,
)
from xdsl.dialects.utils import (
    parse_func_op_like,
    print_func_op_like,
)
from xdsl.ir import (
    Attribute,
    Region,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    IsTerminator,
)

from .attrs import ObjectType
from .encoding import (
    PythonSourceEncodable,
    PythonSourceEncodingContext,
)
from .print import PythonPrintable

##==------------------------------------------------------------------------==##
# Python module
##==------------------------------------------------------------------------==##


class PyOperation(IRDLOperation, PythonSourceEncodable, PythonPrintable):
    pass


@irdl_op_definition
class PyModuleOp(PyOperation):
    """
    Python code is organized into modules and functions.

    Modules are the top-level code.

    Functions are self-explanatory.

    For example if you have the following MLIR:

    py.func @add_two() -> PyObject {
        %0 = py.const 0
        %1 = py.const 1
        %2 = py.binop "add" %0 %1 : PyObject
        py.return %2
    }

    We convert that to the following Python code:
    def add_two():
        _0 = 1
        _1 = 1
        _2 = _0 + _1
        return _2
    """

    name = "py.module"
    body = region_def()

    def dump(self, path: Path):
        ctx = PythonSourceEncodingContext("xdsl-generated")
        with open(path, "w") as fp:
            fp.write("\n".join(self.encode(ctx)))


@irdl_op_definition
class PyFunctionOp(PyOperation):
    name = "py.func"
    sym_name = attr_def(SymbolNameConstraint())
    body = region_def()
    # TODO support parameters later.
    function_type = attr_def(FunctionType)

    def __init__(
        self,
        name: str,
        region: Region,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        attributes: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
        }

        super().__init__(attributes=attributes, regions=[region])

    @classmethod
    def parse(cls, parser: Parser) -> "PyFunctionOp":
        (name, input_types, return_types, region, _, arg_attrs, res_attrs) = (
            parse_func_op_like(
                parser,
                reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
            )
        )
        if arg_attrs:
            raise NotImplementedError("arg_attrs not implemented in arm_func")
        if res_attrs:
            raise NotImplementedError("res_attrs not implemented in arm_func")
        func = cls(name, region, (input_types, return_types))
        return func

    def print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
        )

    def encode(self, ctx: PythonSourceEncodingContext) -> list[str]:
        source: list[str] = []
        source.append(ctx.with_indent(f"def {self.sym_name.data}():\n"))
        with ctx.block():
            for op in self.body.ops:
                if isinstance(op, PyOperation):
                    source.extend(op.encode(ctx))
        return source


@irdl_op_definition
class PyConstOp(PyOperation):
    """
    x = CONST in Python
    """

    name = "py.const"
    assembly_format = "$const attr-dict"

    # We can expand this to other types later.
    const = prop_def(IntegerAttr[I64])
    res = result_def(ObjectType())

    def __init__(self, const: IntegerAttr[I64]):
        super().__init__(properties={"const": const})

    def encode(self, ctx: PythonSourceEncodingContext) -> list[str]:
        return [ctx.with_indent(f"{ctx.get_id()} = {self.const.value.data}\n")]


@irdl_op_definition
class PyBinOp(PyOperation):
    """
    x BINOP y in Python
    """

    name = "py.binop"
    lhs = operand_def(ObjectType())
    rhs = operand_def(ObjectType())
    res = result_def(ObjectType())
    op = prop_def(StringAttr)
    assembly_format = "$op $lhs $rhs attr-dict"

    def __init__(
        self,
    ):
        super().__init__()

    def encode(self, ctx: PythonSourceEncodingContext) -> list[str]:
        match self.op.data:
            case "add":
                kind = "+"
            case _:
                raise NotImplementedError(f"Binop {self.op.data} is not implemented")
        res: list[str] = []
        for op in self.operands:
            if isinstance(op, PyOperation):
                res.extend(op.encode(ctx))
        return res + [
            ctx.with_indent(
                f"{ctx.get_id()} = {ctx.peek_id(3)} {kind} {ctx.peek_id(2)}\n"
            )
        ]


@irdl_op_definition
class PyReturnOp(PyOperation):
    """
    return in Python
    """

    name = "py.return"
    ret = operand_def(ObjectType())
    assembly_format = "$ret attr-dict"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
    ):
        super().__init__()

    def encode(self, ctx: PythonSourceEncodingContext) -> list[str]:
        return [ctx.with_indent(f"return {ctx.peek_id(1)}\n")]
