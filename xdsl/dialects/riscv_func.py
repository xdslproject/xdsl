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
from xdsl.dialects.utils import (
    parse_call_op_like,
    parse_func_op_like,
    parse_return_op_like,
    print_call_op_like,
    print_func_op_like,
    print_return_op_like,
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
    ress: VarOpResult = var_result_def(riscv.IntRegisterType)

    def __init__(
        self,
        callee: SymbolRefAttr,
        args: Sequence[Operation | SSAValue],
        result_types: Sequence[riscv.IntRegisterType],
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
        print_call_op_like(
            printer,
            self,
            self.callee,
            self.args,
            self.attributes,
            reserved_attr_names=("callee",),
        )

    @classmethod
    def parse(cls, parser: Parser) -> CallOp:
        callee, arguments, results, extra_attributes = parse_call_op_like(
            parser, reserved_attr_names=("callee",)
        )
        ress = cast(tuple[riscv.IntRegisterType, ...], results)
        call = CallOp(callee, arguments, ress)
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
        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
        ) = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type")
        )
        func = FuncOp(name, region, (input_types, return_types))
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            reserved_attr_names=("sym_name", "function_type"),
        )


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
        print_return_op_like(printer, self.attributes, self.values)

    @classmethod
    def parse(cls, parser: Parser) -> ReturnOp:
        attrs, args = parse_return_op_like(parser)
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
