from __future__ import annotations

from collections.abc import Sequence

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
    parse_func_op_like,
    print_func_op_like,
)
from xdsl.ir import Attribute, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    opt_attr_def,
    opt_result_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
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
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class SyscallOp(IRDLOperation):
    name = "riscv_func.syscall"
    args = var_operand_def(riscv.IntRegisterType)
    syscall_num = attr_def(IntegerAttr[IntegerType])
    result = opt_result_def(riscv.IntRegisterType)

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
class CallOp(riscv.RISCVInstruction):
    """RISC-V function call operation"""

    name = "riscv_func.call"
    args = var_operand_def(riscv.RISCVRegisterType)
    callee = attr_def(SymbolRefAttr)
    ress = var_result_def(riscv.RISCVRegisterType)

    assembly_format = (
        "$callee `(` $args `)` attr-dict `:` functional-type($args, $ress)"
    )

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

    def assembly_instruction_name(self) -> str:
        return "jal"

    def assembly_line_args(self) -> tuple[riscv.AssemblyInstructionArg | None, ...]:
        return (self.callee.string_value(),)


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
class FuncOp(riscv.RISCVAsmOperation):
    """RISC-V function definition operation"""

    name = "riscv_func.func"
    sym_name = attr_def(StringAttr)
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

        super().__init__(attributes=attributes, regions=[region])

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        visibility = parser.parse_optional_visibility_keyword()
        (name, input_types, return_types, region, extra_attrs, arg_attrs, res_attrs) = (
            parse_func_op_like(
                parser,
                reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
            )
        )
        if arg_attrs:
            raise NotImplementedError("arg_attrs not implemented in riscv_func")
        if res_attrs:
            raise NotImplementedError("res_attrs not implemented in riscv_func")
        func = FuncOp(name, region, (input_types, return_types), visibility)
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        if self.sym_visibility:
            visibility = self.sym_visibility.data
            printer.print(f" {visibility}")

        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
        )

    def assembly_line(self) -> str | None:
        if self.body.blocks:
            return f"{self.sym_name.data}:"
        else:
            return None


@irdl_op_definition
class ReturnOp(riscv.RISCVInstruction):
    """RISC-V function return operation"""

    name = "riscv_func.return"
    values = var_operand_def(riscv.RISCVRegisterType)
    comment = opt_attr_def(StringAttr)

    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    assembly_format = "attr-dict ($values^ `:` type($values))?"

    def __init__(
        self,
        *values: Operation | SSAValue,
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

    def assembly_instruction_name(self) -> str:
        return "ret"

    def assembly_line_args(self) -> tuple[riscv.AssemblyInstructionArg | None, ...]:
        return ()


RISCV_Func = Dialect(
    "riscv_func",
    [
        SyscallOp,
        CallOp,
        FuncOp,
        ReturnOp,
    ],
    [],
)
