from typing import cast

from xdsl.backend.register_type import RegisterType
from xdsl.backend.utils import cast_to_regs
from xdsl.dialects import x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    IndexType,
    ShapedType,
    VectorType,
)
from xdsl.dialects.x86.register import X86VectorRegisterType
from xdsl.ir import Attribute, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import DiagnosticException


def vector_type_to_register_type(
    value_type: VectorType,
    arch: str,
) -> X86VectorRegisterType:
    vector_num_elements = value_type.element_count()
    element_type = cast(FixedBitwidthType, value_type.get_element_type())
    element_size = element_type.bitwidth
    vector_size = vector_num_elements * element_size
    # Choose the x86 vector register according to the
    # target architecture and the abstract vector size
    if vector_size == 128:
        vect_reg_type = x86.register.UNALLOCATED_SSE
    elif vector_size == 256 and (arch == "avx2" or arch == "avx512"):
        vect_reg_type = x86.register.UNALLOCATED_AVX2
    elif vector_size == 512 and arch == "avx512":
        vect_reg_type = x86.register.UNALLOCATED_AVX512
    else:
        raise DiagnosticException(
            "The vector size and target architecture are inconsistent."
        )
    return vect_reg_type


def scalar_type_to_register_type(value_type: Attribute) -> type[RegisterType]:
    assert not isinstance(value_type, ShapedType)
    if (
        isinstance(value_type, FixedBitwidthType) and value_type.bitwidth <= 64
    ) or isinstance(value_type, IndexType):
        return x86.register.GeneralRegisterType
    else:
        raise DiagnosticException("Not implemented for bitwidth larger than 64.")


def cast_operands_to_regs(rewriter: PatternRewriter) -> list[SSAValue[Attribute]]:
    new_ops, new_operands = cast_to_regs(
        values=rewriter.current_operation.operands,
        register_map=scalar_type_to_register_type,
    )
    rewriter.insert_op_before_matched_op(new_ops)
    return new_operands
