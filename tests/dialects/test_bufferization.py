from xdsl.dialects.bufferization import AllocTensorOp
from xdsl.dialects.builtin import TensorType, f64


def test_alloc_tensor_static():
    t = TensorType(f64, [10, 20, 30])
    alloc_tensor = AllocTensorOp.static_type(t)

    assert alloc_tensor.tensor.type == t
    assert alloc_tensor.dynamic_sizes == ()
    assert alloc_tensor.copy is None
    assert alloc_tensor.size_hint is None
