from __future__ import annotations

import itertools
import struct
from collections.abc import Callable, Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

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
from xdsl.utils.bitwise_casts import convert_u32_to_f32
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
class RawPtr:
    """
    Data structure to help simulate pointers into memory.
    """

    memory: bytearray
    offset: int = field(default=0)
    deallocated: bool = field(default=False)

    @staticmethod
    def zeros(count: int) -> RawPtr:
        """
        Returns a new Ptr of size `count` with offset 0.
        """
        return RawPtr(bytearray(count))

    @staticmethod
    def new(el_format: str, els: Sequence[tuple[Any, ...]]) -> RawPtr:
        """
        Returns a new Ptr. The first parameter is a format string as specified in the
        `struct` module, and elements to set.
        """
        el_size = struct.calcsize(el_format)
        res = RawPtr.zeros(len(els) * el_size)
        for i, el in enumerate(els):
            struct.pack_into(el_format, res.memory, i * el_size, *el)
        return res

    def get_iter(self, format: str) -> Iterator[Any]:
        if self.deallocated:
            raise ValueError("Cannot get item of deallocated ptr")
        return (
            values[0]
            for values in struct.iter_unpack(
                format, memoryview(self.memory)[self.offset :]
            )
        )

    def get(self, format: str) -> Any:
        return next(self.get_iter(format))

    def set(self, format: str, *item: Any):
        if self.deallocated:
            raise ValueError("Cannot set item of deallocated ptr")
        struct.pack_into(format, self.memory, self.offset, *item)

    def get_list(self, format: str, count: int):
        return list(itertools.islice(self.get_iter(format), count))

    def deallocate(self) -> None:
        self.deallocated = True

    def __add__(self, offset: int) -> RawPtr:
        """
        Aliases the data, so storing into the offset stores for all other references
        to the list.
        """
        return RawPtr(self.memory, self.offset + offset)

    @property
    def int32(self) -> TypedPtr[int]:
        return TypedPtr(self, ">i")

    @staticmethod
    def new_int32(els: Sequence[int]) -> RawPtr:
        return RawPtr.new(">i", [(el,) for el in els])

    @property
    def float32(self) -> TypedPtr[float]:
        return TypedPtr(self, ">f")

    @staticmethod
    def new_float32(els: Sequence[float]) -> RawPtr:
        return RawPtr.new(">f", [(el,) for el in els])

    @property
    def float64(self) -> TypedPtr[float]:
        return TypedPtr(self, ">d")

    @staticmethod
    def new_float64(els: Sequence[float]) -> RawPtr:
        return RawPtr.new(">d", [(el,) for el in els])


@dataclass
class TypedPtr(Generic[_T]):
    raw: RawPtr
    format: str

    @property
    def size(self) -> int:
        return struct.calcsize(self.format)

    def get_list(self, count: int) -> list[_T]:
        return self.raw.get_list(self.format, count)

    def __getitem__(self, index: int) -> _T | MutableSequence[_T]:
        return (self.raw + index * self.size).get(self.format)

    def __setitem__(self, index: int, value: _T):
        (self.raw + index * self.size).set(self.format, value)


@register_impls
class RiscvFunctions(InterpreterFunctions):
    module_op: ModuleOp
    _data: dict[str, RawPtr] | None
    custom_instructions: dict[str, CustomInstructionFn] = {}
    bitwidth: int

    def __init__(
        self,
        module_op: ModuleOp,
        *,
        bitwidth: int = 32,
        data: dict[str, RawPtr] | None = None,
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
    def get_data(module_op: ModuleOp) -> dict[str, RawPtr]:
        for op in module_op.ops:
            if isinstance(op, riscv.AssemblySectionOp):
                if op.directive.data == ".data":
                    data: dict[str, RawPtr] = {}

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
                                data[label.label.data] = RawPtr.new_int32(ints)
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
    ) -> int | RawPtr:
        match imm:
            case IntegerAttr():
                return imm.value.data
            case riscv.LabelAttr():
                data = self.get_value(op, imm.data)
                return data

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
        if isinstance(imm, RawPtr):
            raise NotImplementedError("Cannot compare pointer in interpreter")
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
        (args[0] + op.immediate.value.data).int32[0] = args[1]
        return ()

    @impl(riscv.LwOp)
    def run_lw(
        self,
        interpreter: Interpreter,
        op: riscv.LwOp,
        args: tuple[Any, ...],
    ):
        offset = self.get_immediate_value(op, op.immediate)
        assert isinstance(offset, int)
        return ((args[0] + offset).int32[0],)

    @impl(riscv.LabelOp)
    def run_label(
        self,
        interpreter: Interpreter,
        op: riscv.LabelOp,
        args: tuple[Any, ...],
    ):
        return ()

    # region F extension

    @impl(riscv.FMulSOp)
    def run_fmul(
        self,
        interpreter: Interpreter,
        op: riscv.FMulSOp,
        args: tuple[Any, ...],
    ):
        return (args[0] * args[1],)

    @impl(riscv.FCvtSWOp)
    def run_fcvt_s_w(
        self,
        interpreter: Interpreter,
        op: riscv.FCvtSWOp,
        args: tuple[Any, ...],
    ):
        return (convert_u32_to_f32(args[0]),)

    @impl(riscv.FSwOp)
    def run_fsw(
        self,
        interpreter: Interpreter,
        op: riscv.FSwOp,
        args: tuple[Any, ...],
    ):
        (args[0] + op.immediate.value.data).float32[0] = args[1]
        return ()

    @impl(riscv.FLwOp)
    def run_flw(
        self,
        interpreter: Interpreter,
        op: riscv.FLwOp,
        args: tuple[Any, ...],
    ):
        offset = self.get_immediate_value(op, op.immediate)
        return ((args[0] + offset).float32[0],)

    @impl(riscv.FSdOp)
    def run_fsd(
        self,
        interpreter: Interpreter,
        op: riscv.FSdOp,
        args: tuple[Any, ...],
    ):
        (args[0] + op.immediate.value.data).float64[0] = args[1]
        return ()

    @impl(riscv.FLdOp)
    def run_fld(
        self,
        interpreter: Interpreter,
        op: riscv.FLdOp,
        args: tuple[Any, ...],
    ):
        offset = self.get_immediate_value(op, op.immediate)
        return ((args[0] + offset).float64[0],)

    # endregion

    @impl(riscv.GetRegisterOp)
    def run_get_register(
        self, interpreter: Interpreter, op: riscv.GetRegisterOp, args: PythonValues
    ) -> PythonValues:
        if not op.res.type == riscv.Registers.ZERO:
            raise InterpretationError(
                f"Cannot interpret riscv.get_register op with non-ZERO type {op.res.type}"
            )

        return (0,)

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
