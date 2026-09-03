from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, cast, overload

from xdsl.backend.arch import Arch
from xdsl.builder import Builder
from xdsl.dialects import asm, ptr, x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    ModuleOp,
    StringAttr,
    VectorType,
)
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    Reg8Type,
    Reg16Type,
    Reg32Type,
    Reg64Type,
    SSERegisterType,
    X86RegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa

ARCH_ATTR_NAME = "x86.arch"
"""
Name of the module attribute recording the target this module is compiled for.

Set once at the top of a pipeline so that passes downstream do not each need
their own `arch` option, in the same spirit as an LLVM module carrying its
target triple.
"""


class X86Arch(Arch):
    VECTOR_TYPES_BY_BITWIDTH: ClassVar[dict[int, type[X86VectorRegisterType]]] = {
        128: SSERegisterType
    }
    """
    Supported vector type for a given vector size.
    """

    @staticmethod
    def name() -> str:
        return "unknown"

    @staticmethod
    def arch_for_name(name: str | None) -> X86Arch:
        if name is None:
            return UNKNOWN
        try:
            return _ARCH_BY_NAME[name]
        except KeyError:
            # Same reason as below: without `from None` the traceback leads with
            # `KeyError: 'sse9'` rather than with the diagnostic.
            raise DiagnosticException(
                f"Unsupported arch {name}. Supported arches are "
                f"{sorted(_ARCH_BY_NAME)}."
            ) from None

    @staticmethod
    def from_module(module: ModuleOp) -> X86Arch:
        """
        Read the target from the module, defaulting to the conservative
        `unknown` target when it is not recorded.
        """
        attr = module.attributes.get(ARCH_ATTR_NAME)
        if attr is None:
            return UNKNOWN
        if not isinstance(attr, StringAttr):
            raise DiagnosticException(
                f"`{ARCH_ATTR_NAME}` must be a string attribute, got {attr}."
            )
        return X86Arch.arch_for_name(attr.data)

    def set_on_module(self, module: ModuleOp) -> None:
        """
        Record this target on the module.
        """
        module.attributes[ARCH_ATTR_NAME] = StringAttr(self.name())

    def default_allocatable_registers(self) -> tuple[X86RegisterType, ...]:
        """
        The registers the allocator may use on this target.

        The upper half of each vector bank, xmm16-31 and ymm16-31, is only
        reachable through EVEX, so it exists on AVX-512 targets and nowhere
        else. Every vector bank shares one allocation pool, so the vector half
        is indexed off a single bank rather than listing each of them.
        """
        return (
            *Reg64Type.allocatable_registers(),
            *AVX2RegisterType.allocatable_registers()[:16],
        )

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
        try:
            return self.VECTOR_TYPES_BY_BITWIDTH[vector_size]
        except KeyError:
            # `from None` keeps the raw `KeyError: 512` out of the traceback, so
            # the reported cause is the diagnostic rather than a dict lookup.
            raise DiagnosticException(
                f"The vector size ({vector_size} bits) and target architecture "
                f"`{self.name()}` are inconsistent. Supported vector sizes are "
                f"{sorted(self.VECTOR_TYPES_BY_BITWIDTH)}."
            ) from None

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
        return [
            builder.insert(
                asm.ToRegOp.get(v, self.register_type_for_type(v.type).unallocated())
            ).register
            for v in values
        ]

    def move_value_to_unallocated(
        self,
        value: SSAValue,
        builder: Builder,
        *,
        value_type: Attribute | None,
        insertion_point: InsertPoint | None = None,
    ) -> SSAValue:
        """
        Move the value to a new register.
        If the value type is known, use a specialised move operation, otherwise use a
        default move operation for the input register.
        """
        if value_type is not None and isa(value_type, VectorType[FixedBitwidthType]):
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
        elif isinstance(reg_type := value.type, X86VectorRegisterType):
            # In the future, we want to be more careful about register types.
            mov_op = x86.ops.DS_VmovapdOp(
                value, destination=type(reg_type).unallocated()
            )
        elif isinstance(reg_type, GeneralRegisterType):
            mov_op = x86.DS_MovOp(value, destination=type(reg_type).unallocated())
        else:
            raise ValueError(f"Invalid type for move {value.type}")

        result = builder.insert(mov_op, insertion_point).results[0]
        result.name_hint = value.name_hint
        return result


UNKNOWN = X86Arch()


class AVX2Arch(X86Arch):
    @staticmethod
    def name() -> str:
        return "avx2"

    VECTOR_TYPES_BY_BITWIDTH = {128: SSERegisterType, 256: AVX2RegisterType}


AVX2 = AVX2Arch()


class AVX512Arch(X86Arch):
    @staticmethod
    def name() -> str:
        return "avx512"

    VECTOR_TYPES_BY_BITWIDTH = {
        128: SSERegisterType,
        256: AVX2RegisterType,
        512: AVX512RegisterType,
    }

    def default_allocatable_registers(self) -> tuple[X86RegisterType, ...]:
        return (
            *Reg64Type.allocatable_registers(),
            *AVX2RegisterType.allocatable_registers(),
        )


AVX512 = AVX512Arch()


_ARCH_BY_NAME = {
    arch_type.name(): arch_type() for arch_type in (X86Arch, AVX2Arch, AVX512Arch)
}
"""
Handled architectures in x86 backend.
"""
