from __future__ import annotations

from dataclasses import dataclass

from xdsl.analysis.dataflow import (
    ChangeResult,
    GenericLatticeAnchor,
)


# region ChangeResult tests
def test_change_result_or():
    assert ChangeResult.CHANGE | ChangeResult.CHANGE is ChangeResult.CHANGE
    assert ChangeResult.CHANGE | ChangeResult.NO_CHANGE is ChangeResult.CHANGE
    assert ChangeResult.NO_CHANGE | ChangeResult.CHANGE is ChangeResult.CHANGE
    assert ChangeResult.NO_CHANGE | ChangeResult.NO_CHANGE is ChangeResult.NO_CHANGE


# endregion


# region GenericLatticeAnchor tests
@dataclass(frozen=True)
class MyAnchor(GenericLatticeAnchor):
    name: str

    def __str__(self) -> str:
        return f"MyAnchor({self.name})"


def test_generic_lattice_anchor():
    anchor1 = MyAnchor("a")
    anchor2 = MyAnchor("a")
    anchor3 = MyAnchor("b")

    assert anchor1 == anchor2
    assert anchor1 != anchor3
    assert hash(anchor1) == hash(anchor2)
    assert hash(anchor1) != hash(anchor3)
    assert str(anchor1) == "MyAnchor(a)"
    assert not (anchor1 == "a")


# endregion
