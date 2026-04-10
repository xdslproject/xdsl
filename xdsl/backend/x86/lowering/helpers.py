from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

from typing_extensions import deprecated

from xdsl.backend.utils import cast_to_regs
from xdsl.builder import Builder
from xdsl.dialects import ptr, x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    VectorType,
)
from xdsl.dialects.x86.registers import (
    GeneralRegisterType,
    Reg8Type,
    Reg16Type,
    Reg32Type,
    Reg64Type,
    X86RegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa
from xdsl.utils.str_enum import StrEnum


class Arch(StrEnum):
    UNKNOWN = "unknown"
    AVX2 = "avx2"
    AVX512 = "avx512"

    @staticmethod
    def arch_for_name(name: str | None) -> Arch:
        if name is None:
            return Arch.UNKNOWN
        try:
            return _ARCH_BY_NAME[name]
        except KeyError:
            raise DiagnosticException(f"Unsupported arch {name}")

    def _register_type_for_vector_type(
        self, value_type: VectorType
    ) -> type[X86VectorRegisterType]:
        """
        Given any vector type, returns the appropriate register type.
        The vector type must fit exactly into a full bitwidth vector supported by the
        ISA, otherwise a `DiagnosticException` is raised.
        """
        vector_num_elements = value_type.element_count()
        element_type = cast(FixedBitwidthType, value_type.get_element_type())
        element_size = element_type.bitwidth
        vector_size = vector_num_elements * element_size
        match self, vector_size:
            case ((Arch.AVX2 | Arch.AVX512), 256):
                return x86.registers.AVX2RegisterType
            case Arch.AVX512, 512:
                return x86.registers.AVX512RegisterType
            case _, 128:
                return x86.registers.SSERegisterType
            case _:
                raise DiagnosticException(
                    f"The vector size ({vector_size} bits) and target architecture `{self}` are inconsistent."
                )

    def _scalar_type_for_type(self, value_type: Attribute) -> type[GeneralRegisterType]:
        if isinstance(value_type, FixedBitwidthType):
            match value_type.bitwidth:
                case 64:
                    return Reg64Type
                case 32:
                    return Reg32Type
                case 16:
                    return Reg16Type
                case 8:
                    return Reg8Type
                case _:
                    ...
        if isinstance(value_type, IndexType) or isinstance(value_type, ptr.PtrType):
            return Reg64Type
        raise DiagnosticException(f"Register type for type {value_type} not supported.")

    @overload
    def register_type_for_type(
        self, value_type: VectorType
    ) -> type[X86VectorRegisterType]: ...

    @overload
    def register_type_for_type(
        self, value_type: Attribute
    ) -> type[X86RegisterType]: ...

    def register_type_for_type(self, value_type: Attribute) -> type[X86RegisterType]:
        if isinstance(value_type, X86RegisterType):
            return type(value_type)
        if isa(value_type, VectorType):
            return self._register_type_for_vector_type(value_type)
        return self._scalar_type_for_type(value_type)

    def cast_to_regs(
        self, values: Sequence[SSAValue], builder: Builder
    ) -> list[SSAValue]:
        return cast_to_regs(values, self.register_type_for_type, builder)

    @deprecated("Please use `arch.cast_to_regs(values, rewriter)`")
    def cast_operands_to_regs(self, rewriter: PatternRewriter) -> list[SSAValue]:
        new_operands = self.cast_to_regs(rewriter.current_operation.operands, rewriter)
        return new_operands

    def move_value_to_unallocated(
        self, value: SSAValue, value_type: Attribute, builder: Builder
    ) -> SSAValue:
        if isa(value_type, VectorType[FixedBitwidthType]):
            if not isinstance(reg_type := value.type, X86VectorRegisterType):
                raise ValueError(f"Invalid type for move {value_type}")
            # Choose the x86 vector instruction according to the
            # abstract vector element size
            match value_type.get_element_type().bitwidth:
                case 16:
                    raise DiagnosticException(
                        "Half-precision floating point vector move is not implemented yet."
                    )
                case 32:
                    raise DiagnosticException(
                        "Half-precision floating point vector move is not implemented yet."
                    )
                case 64:
                    mov_op = x86.ops.DS_VmovapdOp(
                        value, destination=type(reg_type).unallocated()
                    )
                case _:
                    raise DiagnosticException(
                        "Float precision must be half, single or double."
                    )
        else:
            if not isinstance(reg_type := value.type, X86RegisterType):
                raise ValueError(f"Invalid type for move {value_type}")
            mov_op = x86.DS_MovOp(value, destination=type(reg_type).unallocated())

        return builder.insert_op(mov_op).results[0]


_ARCH_BY_NAME = {str(case): case for case in Arch}
"""
Handled architectures in x86 backend.
"""
