from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeAlias, TypeVar, cast

from xdsl.dialects import builtin, riscv
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
)
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    impl_attr,
    impl_cast,
    register_impls,
)
from xdsl.interpreters import ptr
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.ir import Attribute, SSAValue
from xdsl.utils.bitwise_casts import convert_u32_to_f32
from xdsl.utils.comparisons import to_signed, to_unsigned
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
            raise InterpretationError(f"Value not found for register name {name}")

        stored_value = registers[name]

        if stored_value != value:
            raise InterpretationError(
                f"Runtime and stored value mismatch: {value} != {stored_value}"
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
        assert len(results) == len(values)
        return tuple(
            RiscvFunctions.set_reg_value(interpreter, result.type, value)
            for result, value in zip(results, values)
        )

    @staticmethod
    def data(interpreter: Interpreter) -> dict[str, Any]:
        return interpreter.get_data(
            RiscvFunctions,
            _DATA_KEY,
            lambda: RiscvFunctions.get_data(interpreter.module),
        )

    @staticmethod
    def registers(interpreter: Interpreter) -> dict[str, Any]:
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
        self, interpreter: Interpreter, imm: AnyIntegerAttr | riscv.LabelAttr
    ) -> int | ptr.RawPtr:
        match imm:
            case IntegerAttr():
                return imm.value.data
            case riscv.LabelAttr():
                data = self.get_data_value(interpreter, imm.data)
                return data

    @impl(riscv.LiOp)
    def run_li(
        self,
        interpreter: Interpreter,
        op: riscv.LiOp,
        args: tuple[Any, ...],
    ):
        results = (self.get_immediate_value(interpreter, op.immediate),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.MVOp)
    def run_mv(
        self,
        interpreter: Interpreter,
        op: riscv.MVOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = args
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.SltiuOp)
    def run_sltiu(
        self,
        interpreter: Interpreter,
        op: riscv.SltiuOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        unsigned_lhs = to_unsigned(args[0], self.bitwidth)
        imm = self.get_immediate_value(interpreter, op.immediate)
        if isinstance(imm, ptr.RawPtr):
            raise NotImplementedError("Cannot compare pointer in interpreter")
        unsigned_imm = to_unsigned(imm, self.bitwidth)
        results = (int(unsigned_lhs < unsigned_imm),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.SltOp)
    def run_slt(
        self,
        interpreter: Interpreter,
        op: riscv.SltOp,
        args: tuple[Any, ...],
    ):
        signed_lhs = to_signed(args[0], self.bitwidth)
        signed_rhs = to_signed(args[1], self.bitwidth)
        return (int(signed_lhs < signed_rhs),)

    @impl(riscv.AddOp)
    def run_add(
        self,
        interpreter: Interpreter,
        op: riscv.AddOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] + args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.AddiOp)
    def run_addi(
        self,
        interpreter: Interpreter,
        op: riscv.AddiOp,
        args: tuple[Any, ...],
    ):
        immediate = cast(IntegerAttr[IntegerType | IndexType], op.immediate).value.data
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] + immediate,)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.SubOp)
    def run_sub(
        self,
        interpreter: Interpreter,
        op: riscv.SubOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] - args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.SlliOp)
    def run_shift_left(
        self,
        interpreter: Interpreter,
        op: riscv.SlliOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        imm = self.get_immediate_value(interpreter, op.immediate)
        assert isinstance(imm, int)
        results = (args[0] << imm,)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.MulOp)
    def run_mul(
        self,
        interpreter: Interpreter,
        op: riscv.MulOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        lhs = to_signed(args[0], self.bitwidth)
        rhs = to_signed(args[1], self.bitwidth)

        results = (lhs * rhs,)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.SwOp)
    def run_sw(
        self,
        interpreter: Interpreter,
        op: riscv.SwOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        (args[0] + op.immediate.value.data).int32[0] = args[1]
        return ()

    @impl(riscv.LwOp)
    def run_lw(
        self,
        interpreter: Interpreter,
        op: riscv.LwOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        offset = self.get_immediate_value(interpreter, op.immediate)
        assert isinstance(offset, int)
        results = ((args[0] + offset).int32[0],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

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
    def run_fmul_s(
        self,
        interpreter: Interpreter,
        op: riscv.FMulSOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] * args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FMvWXOp)
    def run_fmv_w_x(
        self,
        interpreter: Interpreter,
        op: riscv.FMvWXOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (convert_u32_to_f32(args[0]),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FSwOp)
    def run_fsw(
        self,
        interpreter: Interpreter,
        op: riscv.FSwOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        (args[0] + op.immediate.value.data).float32[0] = args[1]
        return ()

    @impl(riscv.FLwOp)
    def run_flw(
        self,
        interpreter: Interpreter,
        op: riscv.FLwOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        offset = self.get_immediate_value(interpreter, op.immediate)
        assert isinstance(offset, int)
        results = ((args[0] + offset).float32[0],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FMVOp)
    def run_fmv(
        self,
        interpreter: Interpreter,
        op: riscv.FMVOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = args
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    # endregion

    # region D extension

    @impl(riscv.FAddDOp)
    def run_fadd_d(
        self,
        interpreter: Interpreter,
        op: riscv.FAddDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] + args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FSubDOp)
    def run_fsub_d(
        self,
        interpreter: Interpreter,
        op: riscv.FSubDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] - args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FMulDOp)
    def run_fmul_d(
        self,
        interpreter: Interpreter,
        op: riscv.FMulDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] * args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FDivDOp)
    def run_fdiv_d(
        self,
        interpreter: Interpreter,
        op: riscv.FDivDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (args[0] / args[1],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FMinDOp)
    def run_fmin_d(
        self,
        interpreter: Interpreter,
        op: riscv.FMinDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (min(args[0], args[1]),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FMaxDOp)
    def run_fmax_d(
        self,
        interpreter: Interpreter,
        op: riscv.FMaxDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (max(args[0], args[1]),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FCvtDWOp)
    def run_fcvt_d_w(
        self,
        interpreter: Interpreter,
        op: riscv.FCvtDWOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = (float(args[0]),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FSdOp)
    def run_fsd(
        self,
        interpreter: Interpreter,
        op: riscv.FSdOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        offset = op.immediate.value.data
        (args[0] + offset).float64[0] = args[1]
        return ()

    @impl(riscv.FLdOp)
    def run_fld(
        self,
        interpreter: Interpreter,
        op: riscv.FLdOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        offset = self.get_immediate_value(interpreter, op.immediate)
        assert isinstance(offset, int)
        results = ((args[0] + offset).float64[0],)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl(riscv.FMvDOp)
    def run_fmv_d(
        self,
        interpreter: Interpreter,
        op: riscv.FMvDOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        results = args
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    # endregion

    @impl(riscv.GetRegisterOp)
    def run_get_register(
        self, interpreter: Interpreter, op: riscv.GetRegisterOp, args: PythonValues
    ) -> PythonValues:
        attr = op.res.type

        if not isinstance(attr, riscv.RISCVRegisterType):
            raise InterpretationError(f"Unexpected type {attr}, expected register type")

        if not attr.is_allocated:
            raise InterpretationError(
                f"Cannot get value for unallocated register {attr}"
            )

        name = attr.register_name

        registers = RiscvFunctions.registers(interpreter)

        if name not in registers:
            raise InterpretationError(f"Value not found for register name {name}")

        stored_value = registers[name]

        return (stored_value,)

    @impl(riscv.CustomAssemblyInstructionOp)
    def run_custom_instruction(
        self,
        interpreter: Interpreter,
        op: riscv.CustomAssemblyInstructionOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        instr = op.instruction_name.data
        if instr not in self.custom_instructions:
            raise InterpretationError(
                "Could not find custom riscv assembly instruction implementation for"
                f" {instr}"
            )

        results = self.custom_instructions[instr](interpreter, op, args)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)

    @impl_cast(riscv.FloatRegisterType, builtin.Float64Type)
    def cast_float_reg_to_float(
        self,
        input_type: riscv.FloatRegisterType,
        output_type: builtin.Float64Type,
        value: Any,
    ) -> Any:
        return value

    @impl_attr(riscv.IntRegisterType)
    def register_value(
        self,
        interpreter: Interpreter,
        attr: Attribute,
        type_attr: riscv.IntRegisterType,
    ) -> Any:
        match attr:
            case IntegerAttr():
                return attr.value.data
            case builtin.DenseIntOrFPElementsAttr():
                data = [el.value.data for el in attr.data]
                data_ptr = ptr.TypedPtr[Any].new(
                    data,
                    xtype=xtype_for_el_type(
                        attr.get_element_type(), interpreter.index_bitwidth
                    ),
                )
                return data_ptr
            case _:
                interpreter.raise_error(f"Unknown value type for int register: {attr}")
