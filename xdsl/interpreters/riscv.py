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
from xdsl.interpreters.comparisons import to_signed, to_unsigned
from xdsl.ir.core import Operation
from xdsl.utils.exceptions import InterpretationError

_T = TypeVar("_T")


def pairs(els: list[_T]) -> Iterator[tuple[_T, _T]]:
    count = len(els)
    assert not count % 2
    for i in range(0, count, 2):
        yield els[i], els[i + 1]


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
    _data: dict[str, Any] | None
    custom_instructions: dict[str, CustomInstructionFn] = {}
    bitwidth: int

    def __init__(
        self,
        module_op: ModuleOp,
        *,
        bitwidth: int = 32,
        data: dict[str, Any] | None = None,
        custom_instructions: dict[str, CustomInstructionFn] | None = None,
    ):
        super().__init__()
        self.module_op = module_op
        self.bitwidth = bitwidth
        self._data = data
        if custom_instructions is None:
            custom_instructions = {}
        self.custom_instructions = custom_instructions

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            self._data = RiscvFunctions.get_data(self.module_op)
        return self._data

    @staticmethod
    def get_data(module_op: ModuleOp) -> dict[str, Any]:
        for op in module_op.ops:
            if isinstance(op, riscv.AssemblySectionOp):
                if op.directive.data == ".data":
                    data: dict[str, Any] = {}

                    assert op.data is not None
                    ops = list(op.data.block.ops)
                    for label, data_op in pairs(ops):
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
        unsigned_lhs = to_unsigned(args[0], self.bitwidth)
        imm = self.get_immediate_value(op, op.immediate)
        if isinstance(imm, Buffer):
            raise NotImplementedError("Cannot compare buffer pointer in interpreter")
        unsigned_imm = to_unsigned(imm, self.bitwidth)
        return (int(unsigned_lhs < unsigned_imm),)

    @impl(riscv.AddOp)
    def run_add(
        self,
        interpreter: Interpreter,
        op: riscv.AddOp,
        args: tuple[Any, ...],
    ):
        return (args[0] + args[1],)

    @impl(riscv.SlliOp)
    def run_shift_left(
        self,
        interpreter: Interpreter,
        op: riscv.SlliOp,
        args: tuple[Any, ...],
    ):
        imm = self.get_immediate_value(op, op.immediate)
        assert isinstance(imm, int)
        return (args[0] << imm,)

    @impl(riscv.MulOp)
    def run_mul(
        self,
        interpreter: Interpreter,
        op: riscv.MulOp,
        args: tuple[Any, ...],
    ):
        lhs = to_signed(args[0], self.bitwidth)
        rhs = to_signed(args[1], self.bitwidth)

        return (lhs * rhs,)

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
