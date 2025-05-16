from typing import cast

from xdsl.dialects import x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    VectorType,
)
from xdsl.dialects.x86.register import X86VectorRegisterType
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
