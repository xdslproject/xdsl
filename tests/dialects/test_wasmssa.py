import pytest

from xdsl.dialects.wasmssa import (
    AddOp,
    AndOp,
    CopySignOp,
    DivOp,
    DivSIOp,
    DivUIOp,
    EqOp,
    EqzOp,
    GeOp,
    GeSIOp,
    GeUIOp,
    GtOp,
    GtSIOp,
    GtUIOp,
    LeOp,
    LeSIOp,
    LeUIOp,
    LtOp,
    LtSIOp,
    LtUIOp,
    MaxOp,
    MinOp,
    MulOp,
    NeOp,
    OrOp,
    RemSIOp,
    RemUIOp,
    SubOp,
    XOrOp,
)
from xdsl.irdl import IRDLOperation
from xdsl.traits import Commutative, NoMemoryEffect, OpTrait, Pure


@pytest.mark.parametrize(
    "op_type, required_traits, forbidden_traits",
    [
        (AddOp, [Pure(), Commutative()], []),
        (AndOp, [Pure(), Commutative()], []),
        (DivOp, [Pure()], [Commutative()]),
        (DivUIOp, [NoMemoryEffect()], [Pure(), Commutative()]),
        (DivSIOp, [NoMemoryEffect()], [Pure(), Commutative()]),
        (MulOp, [Pure(), Commutative()], []),
        (OrOp, [Pure(), Commutative()], []),
        (SubOp, [Pure()], [Commutative()]),
        (RemUIOp, [NoMemoryEffect()], [Pure(), Commutative()]),
        (RemSIOp, [NoMemoryEffect()], [Pure(), Commutative()]),
        (XOrOp, [Pure(), Commutative()], []),
        (MinOp, [Pure(), Commutative()], []),
        (MaxOp, [Pure(), Commutative()], []),
        (CopySignOp, [Pure()], [Commutative()]),
        (EqOp, [Pure(), Commutative()], []),
        (NeOp, [Pure(), Commutative()], []),
        (LtSIOp, [Pure()], [Commutative()]),
        (LtUIOp, [Pure()], [Commutative()]),
        (LeSIOp, [Pure()], [Commutative()]),
        (LeUIOp, [Pure()], [Commutative()]),
        (GtSIOp, [Pure()], [Commutative()]),
        (GtUIOp, [Pure()], [Commutative()]),
        (GeSIOp, [Pure()], [Commutative()]),
        (GeUIOp, [Pure()], [Commutative()]),
        (LtOp, [Pure()], [Commutative()]),
        (LeOp, [Pure()], [Commutative()]),
        (GtOp, [Pure()], [Commutative()]),
        (GeOp, [Pure()], [Commutative()]),
        (EqzOp, [Pure()], [Commutative()]),
    ],
)
def test_traits(
    op_type: type[IRDLOperation],
    required_traits: list[OpTrait],
    forbidden_traits: list[OpTrait],
):
    for trait in required_traits:
        assert op_type.has_trait(trait)
    for trait in forbidden_traits:
        assert not op_type.has_trait(trait)
