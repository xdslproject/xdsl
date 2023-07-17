from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeAlias, TypeVar

from xdsl.dialects import riscv
from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, ModuleOp
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.ir.core import Operation
from xdsl.utils.exceptions import InterpretationError

_T = TypeVar("_T")


def pairs(it: Iterator[_T]) -> Iterator[tuple[_T, _T]]:
    try:
        while True:
            a = next(it)
            b = next(it)
            yield a, b
    except StopIteration:
        return


CustomInstructionFn: TypeAlias = Callable[
    [Interpreter, riscv.CustomAssemblyInstructionOp, PythonValues], PythonValues
]


@dataclass
class Buffer(Generic[_T]):
    """
    Data structure to help simulate pointer offsets into buffers of data in memory.
    """

    data: list[_T]
    offset: int = 0

    def __add__(self, offset: int) -> Buffer[_T]:
        """
        Aliases the data list, so storing into the offset stores for all other references
        to the list.
        """
        return Buffer(self.data, self.offset + offset // 4)

    def __getitem__(self, key: int) -> _T:
        return self.data[self.offset + key]

    def __setitem__(self, key: int, value: _T):
        self.data[self.offset + key] = value


@register_impls
class RiscvFunctions(InterpreterFunctions):
    module_op: ModuleOp
    data: dict[str, Any]
    custom_instructions: dict[str, CustomInstructionFn] = {}

    def __init__(
        self,
        module_op: ModuleOp,
        *,
        data: dict[str, Any] | None = None,
        custom_instructions: dict[str, CustomInstructionFn] | None = None,
    ):
        super().__init__()
        self.module_op = module_op
        if data is None:
            data = RiscvFunctions.get_data(module_op)
        self.data = data
        if custom_instructions is None:
            custom_instructions = {}
        self.custom_instructions = custom_instructions

    @staticmethod
    def get_data(module_op: ModuleOp) -> dict[str, Any]:
        for op in module_op.ops:
            if isinstance(op, riscv.DirectiveOp):
                if op.directive.data == ".data":
                    data: dict[str, Any] = {}

                    assert op.data is not None
                    ops = list(op.data.block.ops)
                    assert not len(ops) % 2
                    for label, data_op in pairs(iter(ops)):
                        assert isinstance(label, riscv.LabelOp)
                        assert isinstance(data_op, riscv.DirectiveOp)
                        assert data_op.value is not None
                        match data_op.directive.data:
                            case ".word":
                                hexs = data_op.value.data.split(",")
                                ints = [int(hex.strip(), 16) for hex in hexs]
                                data[label.label.data] = ints
                            case _:
                                assert (
                                    False
                                ), f"Unexpected directive {data_op.directive.data}"
                    return data
        else:
            assert False, "Could not find data section"

    def get_value(self, op: Operation, key: str) -> Any:
        return self.data[key]

    def get_immediate_value(
        self, op: Operation, imm: AnyIntegerAttr | riscv.LabelAttr
    ) -> int | Buffer[int]:
        match imm:
            case IntegerAttr():
                return imm.value.data
            case riscv.LabelAttr():
                data = self.get_value(op, imm.data)
                return Buffer(data)

    @impl(riscv.LiOp)
    def run_li(
        self,
        interpreter: Interpreter,
        op: riscv.LiOp,
        args: tuple[Any, ...],
    ):
        return (self.get_immediate_value(op, op.immediate),)

    @impl(riscv.MVOp)
    def run_mv(
        self,
        interpreter: Interpreter,
        op: riscv.MVOp,
        args: tuple[Any, ...],
    ):
        return args

    @impl(riscv.SltiuOp)
    def run_sltiu(
        self,
        interpreter: Interpreter,
        op: riscv.SltiuOp,
        args: tuple[Any, ...],
    ):
        assert len(args) == 2
        if args[0] < args[1]:
            value = self.get_immediate_value(op, op.immediate)
            return (value,)
        else:
            return (args[0],)

    @impl(riscv.SltiOp)
    def run_slti(
        self,
        interpreter: Interpreter,
        op: riscv.SltiOp,
        args: tuple[Any, ...],
    ):
        assert len(args) == 2
        if args[0] < args[1]:
            value = self.get_immediate_value(op, op.immediate)
            return (value,)
        else:
            return (args[0],)

    @impl(riscv.AddOp)
    def run_add(
        self,
        interpreter: Interpreter,
        op: riscv.AddOp,
        args: tuple[Any, ...],
    ):
        return (args[0] + args[1],)

    @impl(riscv.MulOp)
    def run_mul(
        self,
        interpreter: Interpreter,
        op: riscv.MulOp,
        args: tuple[Any, ...],
    ):
        return (args[0] * args[1],)

    @impl(riscv.SwOp)
    def run_sw(
        self,
        interpreter: Interpreter,
        op: riscv.SwOp,
        args: tuple[Any, ...],
    ):
        args[0][op.immediate.value.data] = args[1]
        return ()

    @impl(riscv.LwOp)
    def run_lw(
        self,
        interpreter: Interpreter,
        op: riscv.LwOp,
        args: tuple[Any, ...],
    ):
        offset = self.get_immediate_value(op, op.immediate)
        return (args[0][offset],)

    @impl(riscv.LabelOp)
    def run_label(
        self,
        interpreter: Interpreter,
        op: riscv.LabelOp,
        args: tuple[Any, ...],
    ):
        return ()

    @impl(riscv.GetRegisterOp)
    def run_get_register(
        self,
        interpreter: Interpreter,
        op: riscv.GetRegisterOp,
        args: tuple[Any, ...],
    ):
        if op.res.type == riscv.Registers.ZERO:
            return (0,)
        else:
            raise ValueError(
                "Cannot interpret GetRegisterOp for registers other than ZERO"
            )

    @impl(riscv.CustomAssemblyInstructionOp)
    def run_custom_instruction(
        self,
        interpreter: Interpreter,
        op: riscv.CustomAssemblyInstructionOp,
        args: tuple[Any, ...],
    ):
        instr = op.instruction_name.data
        if instr not in self.custom_instructions:
            raise InterpretationError(
                "Could not find custom riscv assembly instruction implementation for"
                f" {instr}"
            )

        return self.custom_instructions[instr](interpreter, op, args)
