from xdsl.dialects.arm_neon import NeonArrangement
from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type, VectorType


def get_arrangement_from_vec_type(vec_type: VectorType) -> NeonArrangement:
    """
    Function to generate a NeonArrangement for the vector type provided.
    Note we assume full 128-bit registers, althoug the Neon instruction set supports half-full (64-bit) arrangements.
    """
    shape = vec_type.shape
    elem_type = vec_type.element_type

    if len(shape) != 1:
        raise ValueError("Expected 1D shape")

    num_elems = shape.data[0].data

    if isinstance(elem_type, Float16Type):
        if num_elems == 8:
            return NeonArrangement.H
        else:
            raise ValueError(
                f"Invalid number of F16 elements in vector: Expected 8, received {num_elems}"
            )
    elif isinstance(elem_type, Float32Type):
        if num_elems == 4:
            return NeonArrangement.S
        else:
            raise ValueError(
                f"Invalid number of F32 elements in vector: Expected 4, received {num_elems}"
            )
    elif isinstance(elem_type, Float64Type):
        if num_elems == 2:
            return NeonArrangement.D
        else:
            raise ValueError(
                f"Invalid number of F64 elements in vector: Expected 2, received {num_elems}"
            )
    else:
        raise ValueError(f"Unsupported vector type provided: {num_elems} x {elem_type}")
