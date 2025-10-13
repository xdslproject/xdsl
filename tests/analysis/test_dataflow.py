from __future__ import annotations

from xdsl.analysis.dataflow import (
    ChangeResult,
)


# region ChangeResult tests
def test_change_result_or():
    assert ChangeResult.CHANGE | ChangeResult.CHANGE is ChangeResult.CHANGE
    assert ChangeResult.CHANGE | ChangeResult.NO_CHANGE is ChangeResult.CHANGE
    assert ChangeResult.NO_CHANGE | ChangeResult.CHANGE is ChangeResult.CHANGE
    assert ChangeResult.NO_CHANGE | ChangeResult.NO_CHANGE is ChangeResult.NO_CHANGE


# endregion
