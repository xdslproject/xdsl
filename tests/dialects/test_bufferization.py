from xdsl.dialects.bufferization import AllocTensorOp, ToTensorOp
from xdsl.dialects.builtin import MemRefType, TensorType, UnitAttr, f64
from xdsl.dialects.test import TestOp


def test_to_tensor():
    memref_t = MemRefType(f64, [10, 20, 30])
    tensor_t = TensorType(f64, [10, 20, 30])
    memref_v = TestOp(result_types=[memref_t]).res[0]

    to_tensor = ToTensorOp(memref_v)
    assert to_tensor.memref == memref_v
    assert to_tensor.restrict is None
    assert to_tensor.writable is None
    assert to_tensor.tensor.type == tensor_t

    to_tensor = ToTensorOp(memref_v, writable=True, restrict=True)
    assert to_tensor.memref == memref_v
    assert to_tensor.restrict == UnitAttr()
    assert to_tensor.writable == UnitAttr()
    assert to_tensor.tensor.type == tensor_t


def test_alloc_tensor_static():
    t = TensorType(f64, [10, 20, 30])
    alloc_tensor = AllocTensorOp.static_type(t)

    assert alloc_tensor.tensor.type == t
    assert alloc_tensor.dynamic_sizes == ()
    assert alloc_tensor.copy is None
    assert alloc_tensor.size_hint is None
