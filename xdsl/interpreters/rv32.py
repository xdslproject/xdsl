from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeAlias

from typing_extensions import TypeVar

from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    StringAttr,
)
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.utils import ptr
from xdsl.ir import Attribute, SSAValue
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


_DATA_KEY = "data"
REGISTERS_KEY = "registers"
STACK_KEY = "stack"


@register_impls
class RiscvFunctions(InterpreterFunctions):
    custom_instructions: dict[str, CustomInstructionFn] = {}
    bitwidth: int

    def __init__(
        self,
        *,
        bitwidth: int = 32,
        custom_instructions: dict[str, CustomInstructionFn] | None = None,
    ):
        super().__init__()
        self.bitwidth = bitwidth
        if custom_instructions is None:
            custom_instructions = {}
        self.custom_instructions = custom_instructions

    @staticmethod
    def get_reg_value(interpreter: Interpreter, attr: Attribute, value: Any) -> Any:
        if not isinstance(attr, riscv.RISCVRegisterType):
            raise InterpretationError(f"Unexpected type {attr}, expected register type")

        if not attr.is_allocated:
            return value

        name = attr.register_name

        registers = RiscvFunctions.registers(interpreter)

        if name not in registers:
            raise InterpretationError(f"Value not found for register name {name.data}")

        stored_value = registers[name]

        if stored_value != value:
            raise InterpretationError(
                f"Runtime and stored value mismatch: {value} != {stored_value} {attr}"
            )

        return value

    @staticmethod
    def set_reg_value(interpreter: Interpreter, attr: Attribute, value: Any) -> Any:
        if not isinstance(attr, riscv.RISCVRegisterType):
            raise InterpretationError(f"Unexpected type {attr}, expected register type")

        if not attr.is_allocated:
            return value

        name = attr.register_name

        if name == riscv.Registers.ZERO.register_name:
            # Values assigned to ZERO are erased
            return 0

        registers = RiscvFunctions.registers(interpreter)

        registers[name] = value

        return value

    @staticmethod
    def get_reg_values(
        interpreter: Interpreter,
        ssa_values: Sequence[SSAValue],
        python_values: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(ssa_values) == len(python_values)
        return tuple(
            RiscvFunctions.get_reg_value(interpreter, ssa_value.type, python_value)
            for ssa_value, python_value in zip(ssa_values, python_values)
        )

    @staticmethod
    def set_reg_values(
        interpreter: Interpreter, results: Sequence[SSAValue], values: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return tuple(
            RiscvFunctions.set_reg_value(interpreter, result.type, value)
            for result, value in zip(results, values, strict=True)
        )

    @staticmethod
    def set_reg_values_for_types(
        interpreter: Interpreter,
        result_types: Sequence[Attribute],
        values: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        return tuple(
            RiscvFunctions.set_reg_value(interpreter, result_type, value)
            for result_type, value in zip(result_types, values, strict=True)
        )

    @staticmethod
    def data(interpreter: Interpreter) -> dict[str, Any]:
        return interpreter.get_data(
            RiscvFunctions,
            _DATA_KEY,
            lambda: RiscvFunctions.get_data(interpreter.module),
        )

    @staticmethod
    def registers(interpreter: Interpreter) -> dict[StringAttr, Any]:
        return interpreter.get_data(
            RiscvFunctions,
            REGISTERS_KEY,
            lambda: {
                riscv.Registers.ZERO.register_name: 0,
                riscv.Registers.SP.register_name: RiscvFunctions.stack(interpreter),
            },
        )

    @staticmethod
    def stack(interpreter: Interpreter) -> ptr.RawPtr:
        """
        Stack memory, by default 1mb.
        """
        stack_size = 1 << 20
        return interpreter.get_data(
            RiscvFunctions,
            STACK_KEY,
            lambda: ptr.RawPtr(bytearray(stack_size), offset=stack_size),
        )

    @staticmethod
    def get_data(module_op: ModuleOp) -> dict[str, ptr.RawPtr]:
        data: dict[str, ptr.RawPtr] = {}
        for op in module_op.ops:
            if isinstance(op, riscv.AssemblySectionOp):
                if op.directive.data == ".data":
                    assert op.data is not None
                    ops = list(op.data.block.ops)
                    for label, data_op in pairs(ops):
                        if not (
                            isinstance(label, riscv.LabelOp)
                            and isinstance(data_op, riscv.DirectiveOp)
                        ):
                            raise InterpretationError(
                                "Interpreter assumes that data section is comprised of "
                                "labels followed by directives"
                            )
                        if data_op.value is None:
                            raise InterpretationError(
                                "Unexpected None value in data section directive"
                            )

                        match data_op.directive.data:
                            case ".word":
                                hexs = data_op.value.data.split(",")
                                ints = [int(hex.strip(), 16) for hex in hexs]
                                data[label.label.data] = ptr.TypedPtr.new_int32(
                                    ints
                                ).raw
                            case _:
                                raise InterpretationError(
                                    "Cannot interpret data directive "
                                    f"{data_op.directive.data}"
                                )
        return data

    def get_data_value(self, interpreter: Interpreter, key: str) -> Any:
        data = self.data(interpreter)
        if key not in data:
            raise InterpretationError(f"No data found for key ({key})")
        return data[key]

    def get_immediate_value(
        self, interpreter: Interpreter, imm: IntegerAttr | riscv.LabelAttr
    ) -> int | ptr.RawPtr:
        match imm:
            case IntegerAttr():
                return imm.value.data
            case riscv.LabelAttr():
                data = self.get_data_value(interpreter, imm.data)
                return data

    @impl(rv32.SlliOp)
    def run_shift_left_i(
        self,
        interpreter: Interpreter,
        op: rv32.SlliOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        imm = self.get_immediate_value(interpreter, op.immediate)
        assert isinstance(imm, int)
        results = (args[0] << imm,)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)
