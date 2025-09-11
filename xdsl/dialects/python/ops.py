"""
This file contains the definition of the WebAssembly (wasm) dialect.

The purpose of this dialect is to model WebAssembly modules, as per the
WebAssembly Specification 2.0.

The paragraphs prefixed by `wasm>` in the documentation of this dialect are
excerpts from the WebAssembly Specification, which is licensed under the terms
at the bottom of this file.
"""

import importlib
from collections.abc import Sequence
from opcode import (
    _inline_cache_entries,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    _nb_ops,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    opmap,
)
from typing import BinaryIO, ClassVar

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
from .bytecode_print import BytecodePrintable, BytecodePrinter
from .encoding import (
    PythonBinaryEncodable,
    PythonBinaryEncodingContext,
    PythonCodeObjectEncodingContext,
)

##==------------------------------------------------------------------------==##
# Python module
##==------------------------------------------------------------------------==##

# Search CPython source for what these mean.
OPMAP: dict[str, int] = opmap
INLINE_CACHE_ENTRIES: dict[str, int] = _inline_cache_entries  # pyright: ignore[reportUnknownVariableType]


NB_OPS: tuple[str, ...] = tuple(op[1] for op in _nb_ops)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]

CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200


class PythonOperation(IRDLOperation, PythonBinaryEncodable, BytecodePrintable):
    pass


@irdl_op_definition
class PythonFunctionOp(PythonOperation):
    """
    Python code is organized into modules and functions.

    Modules are the top-level code.

    Functions are self-explanatory.
    """

    name = "python.func"
    sym_name = attr_def(SymbolNameConstraint())
    body = region_def()
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
    def parse(cls, parser: Parser) -> "PythonFunctionOp":
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

    def encode(self, ctx: PythonBinaryEncodingContext) -> list[int]:
        linear_insns: list[int] = []
        for op in self.body.ops:
            if isinstance(op, PythonOperation):
                linear_insns.extend(op.encode(ctx))
        return linear_insns

    def dump_to_file_as_single_module(self, io: BinaryIO) -> None:
        # See https://stackoverflow.com/questions/73439775/how-to-convert-marshall-code-object-to-pyc-file
        # https://github.com/python/cpython/blob/1ce05537a3ebaf1e5c54505b2272d61bb6cf5de0/Lib/importlib/_bootstrap_external.py#L526
        # Python co_consts always starts with None
        ctx = PythonBinaryEncodingContext(name=f"{self.sym_name}", co_consts=[None])
        linear_insns = self.encode(ctx)
        co_ctx = PythonFunctionOp.compute_ctx(ctx, linear_insns)
        func_code = co_ctx.to_code_object()
        module_ctx = PythonBinaryEncodingContext(
            name="<xdsl.toplevel>", co_consts=[None, func_code]
        )
        module_linear_insns: list[int] = [
            opmap["LOAD_CONST"],
            1,
            opmap["MAKE_FUNCTION"],
            0,
            opmap["STORE_NAME"],
            0,
            opmap["LOAD_CONST"],
            0,
            opmap["RETURN_VALUE"],
            0,
        ]
        co_module_ctx = PythonFunctionOp.compute_ctx(module_ctx, module_linear_insns)
        io.write(
            importlib._bootstrap_external._code_to_timestamp_pyc(  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAttributeAccessIssue]
                co_module_ctx.to_code_object()
            )
        )

    @staticmethod
    def compute_ctx(
        binary_ctx: PythonBinaryEncodingContext, linear_insns: list[int]
    ) -> PythonCodeObjectEncodingContext:
        """
        For example if you have the following MLIR:

        python.func @add_two() -> PyObject {
            %0 = python.load_const 0 : PyObject
            %1 = python.load_const 1 : PyObject
            %2 = python.binary_op add %0 %1 : PyObject
            python.return_value %2
        }

        CPython is a stack machine. So we need to calculate a bunch of properties:
        After calculation we convert it to:

        LOAD_CONST 0
        LOAD_CONST 1
        BINARY_OP (add)
        RETURN_VALUE
        """
        return PythonCodeObjectEncodingContext(
            co_argcount=0,
            co_posonlyargcount=0,
            co_kwonlyargcount=0,
            co_nlocals=3,  # TODO compute me
            # TODO: We need to traverse all BBs and determine the maximum
            # stack depth via backtracking.
            # For now, we just assume one BB.
            # See stack_effect function from opcode (builtin)
            co_stacksize=5,
            co_flags=CO_OPTIMIZED,  # TODO compute me
            co_code=bytes(linear_insns),
            co_consts=tuple(binary_ctx.co_consts),
            co_names=("add_two", "add_two", "add_two"),
            co_varnames=("optimized_away", "optimized_away", "optimized_away"),
            co_filename="<xdsl-binary>",
            co_name=f"{binary_ctx.name}",
            co_firstlineno=0,
            co_cellvars=(),
            co_freevars=(),
            co_qualname=f"<xdsl-binary>.{binary_ctx.name}",
            co_linetable=b"",
            co_exceptiontable=b"",
            co_lnotab="",
        )

    def print_python(self, printer: BytecodePrinter) -> None:
        printer.print_string("module")


@irdl_op_definition
class LoadConstOp(PythonOperation):
    """
    LOAD_CONST in Python
    """

    PYTHON_OPCODE: ClassVar[str] = "LOAD_CONST"
    name = "python.load_const"
    assembly_format = "$const attr-dict"

    const = prop_def(IntegerAttr[I64])
    res = result_def(ObjectType())

    def __init__(self, const: IntegerAttr[I64]):
        super().__init__(properties={"const": const})

    def encode(self, ctx: PythonBinaryEncodingContext) -> list[int]:
        index = len(ctx.co_consts)
        ctx.co_consts.append(self.const.value.data)
        return [opmap[self.PYTHON_OPCODE], index]


BINARY_OP_ALLOWED_OPS: dict[str, str] = {
    "add": "+",
    # TODO: finish the rest.
}


@irdl_op_definition
class BinaryOpOp(PythonOperation):
    """
    BINARY_OP in Python
    """

    PYTHON_OPCODE: ClassVar[str] = "BINARY_OP"
    name = "python.binary_op"
    lhs = operand_def(ObjectType())
    rhs = operand_def(ObjectType())
    res = result_def(ObjectType())
    op = prop_def(StringAttr)
    assembly_format = "$op $lhs $rhs attr-dict"

    def __init__(
        self,
    ):
        super().__init__()

    def encode(self, ctx: PythonBinaryEncodingContext) -> list[int]:
        res: list[int] = [
            opmap[self.PYTHON_OPCODE],
            NB_OPS.index(BINARY_OP_ALLOWED_OPS[self.op.data]),
        ]
        res += [OPMAP["CACHE"], 0] * INLINE_CACHE_ENTRIES[self.PYTHON_OPCODE]
        return res


@irdl_op_definition
class ReturnValueOp(PythonOperation):
    """
    RETURN_VALUE in Python
    """

    PYTHON_OPCODE: ClassVar[str] = "RETURN_VALUE"
    name = "python.return_value"
    ret = operand_def(ObjectType())
    assembly_format = "$ret attr-dict"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
    ):
        super().__init__()

    def encode(self, ctx: PythonBinaryEncodingContext) -> list[int]:
        return [opmap[self.PYTHON_OPCODE], 0]
