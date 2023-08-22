from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    FunctionType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Attribute, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    OptOpResult,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_op_definition,
    opt_attr_def,
    opt_result_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser.core import Parser
from xdsl.printer import Printer
from xdsl.traits import CallableOpInterface, HasParent, IsTerminator, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_op_definition
class SyscallOp(IRDLOperation):
    name = "riscv_func.syscall"
    args: VarOperand = var_operand_def(riscv.IntRegisterType)
    syscall_num: IntegerAttr[IntegerType] = attr_def(IntegerAttr[IntegerType])
    result: OptOpResult = opt_result_def(riscv.IntRegisterType)

    def __init__(
        self,
        num: int | AnyIntegerAttr,
        has_result: bool = False,
        operands: list[SSAValue | Operation] = [],
    ):
        if isinstance(num, int):
            num = IntegerAttr.from_int_and_width(num, 32)
        super().__init__(
            operands=[operands],
            attributes={"syscall_num": num},
            result_types=[riscv.IntRegisterType.unallocated() if has_result else None],
        )

    def verify_(self):
        if len(self.args) >= 7:
            raise VerifyException(
                f"Syscall op has too many operands ({len(self.args)}), expected fewer than 7"
            )
        if len(self.results) >= 3:
            raise VerifyException(
                f"Syscall op has too many results ({len(self.results)}), expected fewer than 3"
            )


@irdl_op_definition
class CallOp(IRDLOperation):
    """RISC-V function call operation"""

    name = "riscv_func.call"
    args: VarOperand = var_operand_def(riscv.IntRegisterType)
    callee: SymbolRefAttr = attr_def(SymbolRefAttr)
    ress: VarOpResult = var_result_def(riscv.RISCVRegisterType)

    def __init__(
        self,
        callee: SymbolRefAttr,
        args: Sequence[Operation | SSAValue],
        result_types: Sequence[riscv.RISCVRegisterType],
        comment: StringAttr | None = None,
    ):
        super().__init__(
            operands=[args],
            result_types=[result_types],
            attributes={
                "callee": callee,
                "comment": comment,
            },
        )

    def verify_(self):
        if len(self.args) >= 9:
            raise VerifyException(
                f"Function op has too many operands ({len(self.args)}), expected fewer than 9"
            )

        if len(self.results) >= 3:
            raise VerifyException(
                f"Function op has too many results ({len(self.results)}), expected fewer than 3"
            )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_attribute(self.callee)
        printer.print_string("(")
        printer.print_list(self.args, printer.print_ssa_value)
        printer.print_string(")")
        printer.print_op_attributes(self.attributes, reserved_attr_names=("callee",))
        printer.print_string(" : ")
        printer.print_operation_type(self)

    @classmethod
    def parse(cls, parser: Parser) -> CallOp:
        callee = parser.parse_symbol_name()
        unresolved_arguments = parser.parse_op_args_list()
        extra_attributes = parser.parse_optional_attr_dict_with_reserved_attr_names(
            ("callee",)
        )
        parser.parse_characters(":")
        pos = parser.pos
        function_type = parser.parse_function_type()
        arguments = parser.resolve_operands(
            unresolved_arguments, function_type.inputs.data, pos
        )
        for attr in function_type.outputs.data:
            if not isinstance(attr, riscv.IntRegisterType):
                parser.raise_error(
                    "Expected register type when parsing riscv_func.call type"
                )
        ress = cast(tuple[riscv.IntRegisterType, ...], function_type.outputs.data)
        call = CallOp(SymbolRefAttr(callee), arguments, ress)
        if extra_attributes is not None:
            call.attributes |= extra_attributes.data
        return call


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body


@irdl_op_definition
class FuncOp(IRDLOperation):
    """RISC-V function definition operation"""

    name = "riscv_func.func"
    sym_name: StringAttr = attr_def(StringAttr)
    body: Region = region_def()
    function_type: FunctionType = attr_def(FunctionType)

    traits = frozenset([SymbolOpInterface(), FuncOpCallableInterface()])

    def __init__(
        self,
        name: str,
        region: Region,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
        }

        super().__init__(attributes=attributes, regions=[region])

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        # Parse function name
        name = parser.parse_symbol_name().data

        def parse_fun_input():
            ret = parser.parse_optional_argument()
            if ret is None:
                ret = parser.parse_optional_type()
            if ret is None:
                parser.raise_error("Expected argument or type")
            return ret

        # Parse function arguments
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN,
            parse_fun_input,
        )

        # Check consistency (They should be either all named or none)
        if isa(args, list[parser.Argument]):
            entry_args = args
            input_types = cast(list[Attribute], [a.type for a in args])
        elif isa(args, list[Attribute]):
            entry_args = None
            input_types = args
        else:
            parser.raise_error(
                "Expected all arguments to be named or all arguments to be unnamed."
            )

        # Parse return type
        if parser.parse_optional_punctuation("->"):
            return_types = parser.parse_optional_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )
            if return_types is None:
                return_types = [parser.parse_type()]
        else:
            return_types = []

        attr_dict = parser.parse_optional_attr_dict_with_keyword(
            (
                "sym_name",
                "function_type",
            )
        )

        # Parse body
        region = parser.parse_optional_region(entry_args)
        if region is None:
            region = Region()
        func = FuncOp(name, region, (input_types, return_types))
        if attr_dict is not None:
            func.attributes |= attr_dict.data
        return func

    def print(self, printer: Printer):
        reserved = {"sym_name", "function_type"}

        printer.print(f" @{self.sym_name.data}")
        if len(self.body.blocks) > 0:
            printer.print("(")
            printer.print_list(self.body.blocks[0].args, printer.print_block_argument)
            printer.print(") ")
            if self.function_type.outputs:
                printer.print("-> ")
                if len(self.function_type.outputs) > 1:
                    printer.print("(")
                printer.print_list(self.function_type.outputs, printer.print_attribute)
                if len(self.function_type.outputs) > 1:
                    printer.print(")")
                printer.print(" ")
        else:
            printer.print_attribute(self.function_type)
        printer.print_op_attributes(
            self.attributes, reserved_attr_names=reserved, print_keyword=True
        )

        if len(self.body.blocks) > 0:
            printer.print_region(self.body, False, False)


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """RISC-V function return operation"""

    name = "riscv_func.return"
    values: VarOperand = var_operand_def(riscv.IntRegisterType)
    comment: StringAttr | None = opt_attr_def(StringAttr)

    traits = frozenset([IsTerminator(), HasParent(FuncOp)])

    def __init__(
        self,
        values: Sequence[Operation | SSAValue],
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[values],
            attributes={
                "comment": comment,
            },
        )

    def verify_(self):
        if len(self.results) >= 3:
            raise VerifyException(
                f"Function op has too many results ({len(self.results)}), expected fewer than 3"
            )

    def print(self, printer: Printer):
        if self.attributes:
            printer.print(" ")
            printer.print_op_attributes(self.attributes)

        if self.values:
            printer.print(" ")
            printer.print_list(self.values, printer.print_ssa_value)
            printer.print_string(" : ")

            printer.print_list(
                (value.type for value in self.values), printer.print_attribute
            )

    @classmethod
    def parse(cls, parser: Parser) -> ReturnOp:
        attrs = parser.parse_optional_attr_dict()

        pos = parser.pos
        unresolved_operands = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )

        args: Sequence[SSAValue]
        if unresolved_operands is not None:
            parser.parse_punctuation(":")
            types = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parser.parse_type, "Expected return value type"
            )
            args = parser.resolve_operands(unresolved_operands, types, pos)
        else:
            args = []

        op = ReturnOp(args)
        op.attributes.update(attrs)
        return op


RISCV_Func = Dialect(
    [
        SyscallOp,
        CallOp,
        FuncOp,
        ReturnOp,
    ],
    [],
)
