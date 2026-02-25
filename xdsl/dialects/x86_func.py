from __future__ import annotations

from collections.abc import Sequence

from xdsl.backend.assembly_printer import AssemblyPrintable, AssemblyPrinter
from xdsl.dialects.builtin import (
    FunctionType,
    LocationAttr,
    StringAttr,
    SymbolNameConstraint,
)
from xdsl.dialects.utils import (
    parse_func_op_like,
    print_func_op_like,
)
from xdsl.dialects.x86.ops import X86Instruction
from xdsl.ir import Attribute, Dialect, Operation, Region
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    opt_attr_def,
    region_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    CallableOpInterface,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import DiagnosticException


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body

    @classmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation, AssemblyPrintable):
    """x86 function definition operation"""

    name = "x86_func.func"
    sym_name = attr_def(SymbolNameConstraint())
    body = region_def()
    function_type = attr_def(FunctionType)
    sym_visibility = opt_attr_def(StringAttr)

    traits = traits_def(
        SymbolOpInterface(),
        FuncOpCallableInterface(),
        IsolatedFromAbove(),
    )

    def __init__(
        self,
        name: str,
        region: Region,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        visibility: StringAttr | str | None = None,
        location: LocationAttr | None = None,
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if isinstance(visibility, str):
            visibility = StringAttr(visibility)
        attributes: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
            "sym_visibility": visibility,
        }

        super().__init__(attributes=attributes, regions=[region], location=location)

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        visibility = parser.parse_optional_visibility_keyword()
        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
            res_attrs,
            location,
        ) = parse_func_op_like(
            parser,
            reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
        )
        if arg_attrs:
            raise NotImplementedError("arg_attrs not implemented in x86_func")
        if res_attrs:
            raise NotImplementedError("res_attrs not implemented in x86_func")
        func = FuncOp(
            name, region, (input_types, return_types), visibility, location=location
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        if self.sym_visibility:
            visibility = self.sym_visibility.data
            printer.print_string(f" {visibility}")

        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
            location=self.get_loc(),
        )

    def print_assembly(self, printer: AssemblyPrinter) -> None:
        if not self.body.blocks:
            # Print nothing for function declaration
            return

        printer.emit_section(".text")

        if self.sym_visibility is not None:
            match self.sym_visibility.data:
                case "public":
                    printer.print_string(f".globl {self.sym_name.data}\n", indent=0)
                case "private":
                    printer.print_string(f".local {self.sym_name.data}\n", indent=0)
                case _:
                    raise DiagnosticException(
                        f"Unexpected visibility {self.sym_visibility.data} for function {self.sym_name}"
                    )

        printer.print_string(f"{self.sym_name.data}:\n")


@irdl_op_definition
class RetOp(X86Instruction):
    """
    Return from subroutine.
    """

    name = "x86_func.ret"

    assembly_format = "attr-dict"

    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(
        self,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            attributes={
                "comment": comment,
            },
        )

    def assembly_line_args(self):
        return ()


X86_FUNC = Dialect(
    "x86_func",
    [
        FuncOp,
        RetOp,
    ],
)
