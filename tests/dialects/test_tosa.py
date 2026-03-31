from xdsl.dialects.builtin import (
    IntegerAttr,
    TensorType,
    f32,
    i32,
)
from xdsl.dialects.tosa import ConcatOp, are_tosa_broadcastable
from xdsl.utils.test_value import create_ssa_value

t_i = TensorType(i32, [1, 2, 3, 4])
t_f = TensorType(f32, [1, 2, 3, 4])
t_flat = TensorType(i32, [1, 1, 1, 1])
t_small = TensorType(i32, [1, 2])
t_large = TensorType(i32, [4, 5, 6, 7])

tensor_i = create_ssa_value(t_i)
tensor_f = create_ssa_value(t_f)
tensor_flat = create_ssa_value(t_flat)
tensor_small = create_ssa_value(t_small)
tensor_large = create_ssa_value(t_large)


def test_are_tosa_broadcastable():
    # test same
    assert are_tosa_broadcastable(t_i, t_i, t_i)

    # test broadcasting
    assert are_tosa_broadcastable(t_i, t_flat, t_i)
    assert are_tosa_broadcastable(t_flat, t_i, t_i)
    assert not are_tosa_broadcastable(t_i, t_flat, t_flat)

    # test shape mismatch
    assert not are_tosa_broadcastable(t_i, t_small, t_i)

    # test mismatched dim sizes
    assert not are_tosa_broadcastable(t_i, t_i, t_large)


def test_init_tosa_concat():
    t_out = TensorType(i32, [1, 4, 3, 4])
    concat = ConcatOp([tensor_i, tensor_i], IntegerAttr(1, i32), t_out)
    assert concat.tensors == (tensor_i, tensor_i)
