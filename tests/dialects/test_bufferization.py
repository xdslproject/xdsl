from typing import ClassVar

import pytest

from xdsl.dialects.bufferization import (
    AllocTensorOp,
    CloneOp,
    TensorFromMemRefConstraint,
    ToTensorOp,
)
from xdsl.dialects.builtin import (
    AnyUnrankedMemRefTypeConstr,
    IndexType,
    IntegerType,
    MemRefType,
    TensorType,
    UnitAttr,
    UnrankedMemRefType,
    UnrankedTensorType,
    f64,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute
from xdsl.irdl import (
    ConstraintContext,
    EqAttrConstraint,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
)
from xdsl.utils.exceptions import VerifyException


def test_tensor_from_memref_inference():
    constr = TensorFromMemRefConstraint(MemRefType.constr())
    assert not constr.can_infer(set())

    constr2 = TensorFromMemRefConstraint(
        EqAttrConstraint(MemRefType(f64, [10, 20, 30]))
    )
    assert constr2.can_infer(set())
    assert constr2.infer(ConstraintContext()) == TensorType(f64, [10, 20, 30])

    constr3 = TensorFromMemRefConstraint(
        EqAttrConstraint(UnrankedMemRefType.from_type(f64))
    )
    assert constr3.can_infer(set())
    assert constr3.infer(ConstraintContext()) == UnrankedTensorType(f64)


@irdl_op_definition
class TensorFromMemRefOp(IRDLOperation):
    name = "test.tensor_from_memref"
    T: ClassVar = VarConstraint("T", MemRefType.constr() | AnyUnrankedMemRefTypeConstr)

    in_tensor = operand_def(
        TensorFromMemRefConstraint(
            MemRefType.constr(element_type=EqAttrConstraint(IndexType()))
        )
    )

    in_var_memref = operand_def(T)

    in_var_tensor = operand_def(TensorFromMemRefConstraint(T))


def test_tensor_from_memref_constraint():
    [v_memref, v_tensor] = TestOp(
        result_types=[
            MemRefType(IndexType(), [10, 20, 30]),
            TensorType(IndexType(), [10, 20, 30]),
        ]
    ).res
    op1 = TensorFromMemRefOp(operands=(v_tensor, v_memref, v_tensor))
    op1.verify()

    [v_unranked_memref, v_unranked_tensor] = TestOp(
        result_types=[
            UnrankedMemRefType.from_type(IndexType()),
            UnrankedTensorType(IndexType()),
        ]
    ).res
    op2 = TensorFromMemRefOp(operands=(v_tensor, v_unranked_memref, v_unranked_tensor))
    op2.verify()


@pytest.mark.parametrize(
    "type1, type2, type3, error",
    [
        (
            IndexType(),
            MemRefType(IndexType(), [10, 20, 30]),
            TensorType(IndexType(), [10, 20, 30]),
            "Expected tensor or unranked tensor type, got index",
        ),
        (
            TensorType(IntegerType(32), [10, 10, 10]),
            MemRefType(IndexType(), [10, 20, 30]),
            TensorType(IndexType(), [10, 20, 30]),
            "Expected attribute index but got i32",
        ),
        (
            UnrankedTensorType(IndexType()),
            MemRefType(IndexType(), [10, 20, 30]),
            TensorType(IndexType(), [10, 20, 30]),
            "memref<\\*xindex> should be of base attribute memref",
        ),
        (
            TensorType(IndexType(), [10, 10, 10]),
            MemRefType(IndexType(), [10, 20, 30]),
            TensorType(IndexType(), [10, 20, 20]),
            "attribute memref<10x20x30xindex> expected from variable 'T', but got memref<10x20x20xindex>",
        ),
        (
            TensorType(IndexType(), [10, 10, 10]),
            MemRefType(IntegerType(32), [10, 20, 30]),
            TensorType(IndexType(), [10, 20, 30]),
            "attribute memref<10x20x30xi32> expected from variable 'T', but got memref<10x20x30xindex>",
        ),
    ],
)
def test_tensor_from_memref_constraint_failure(
    type1: Attribute, type2: Attribute, type3: Attribute, error: str
):
    [v1, v2, v3] = TestOp(
        result_types=[
            type1,
            type2,
            type3,
        ]
    ).res

    op1 = TensorFromMemRefOp(operands=(v1, v2, v3))
    with pytest.raises(VerifyException, match=error):
        op1.verify()


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
    alloc_tensor = AllocTensorOp(t)

    assert alloc_tensor.tensor.type == t
    assert alloc_tensor.dynamic_sizes == ()
    assert alloc_tensor.copy is None
    assert alloc_tensor.size_hint is None


def test_clone():
    memref_t = MemRefType(f64, [10, 20, 30])
    memref_v = TestOp(result_types=[memref_t]).res[0]

    clone = CloneOp(memref_v)

    assert clone.input == memref_v
    assert clone.input.type == memref_t
    assert clone.output.type == memref_t
