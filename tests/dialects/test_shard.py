from collections.abc import Callable

import pytest

from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    i64,
)
from xdsl.dialects.shard import (
    GridOp,
    NeighborsLinearIndicesOp,
    ProcessLinearIndexOp,
    ProcessMultiIndexOp,
)
from xdsl.dialects.test import TestOp
from xdsl.utils.exceptions import VerifyException


def _grid() -> GridOp:
    return GridOp.create(
        properties={
            "sym_name": StringAttr("grid"),
            "shape": DenseArrayBase.from_list(i64, [2, 3]),
        }
    )


def _grid_str():
    return "grid"


def _grid_ref():
    return SymbolRefAttr("grid")


GridRefFactory = Callable[[], GridOp | str | SymbolRefAttr]

GRID_REFS: list[GridRefFactory] = [_grid, _grid_str, _grid_ref]


@pytest.mark.parametrize("make_ref", GRID_REFS)
def test_process_linear_index(make_ref: GridRefFactory):
    op = ProcessLinearIndexOp(make_ref())
    assert op.grid == SymbolRefAttr("grid")
    assert op.result.type == IndexType()
    ModuleOp([_grid(), op]).verify()


@pytest.mark.parametrize(
    "make_ref, axes, num_results",
    [
        (_grid, (), 2),
        (_grid, (1,), 1),
        (_grid_str, (1, 0), 2),
        (_grid_ref, (0,), 1),
    ],
)
def test_process_multi_index(
    make_ref: GridRefFactory, axes: tuple[int, ...], num_results: int
):
    op = ProcessMultiIndexOp(make_ref(), axes)
    assert op.grid == SymbolRefAttr("grid")
    assert op.axes.get_values() == axes
    assert op.result.types == (IndexType(),) * num_results
    ModuleOp([_grid(), op]).verify()


def test_process_multi_index_by_name_no_axes():
    # Without the grid, the number of results cannot be inferred.
    op = ProcessMultiIndexOp("grid")
    assert not op.result
    with pytest.raises(
        VerifyException, match=r"Unexpected number of results 0\. Expected 2\."
    ):
        ModuleOp([_grid(), op]).verify()


@pytest.mark.parametrize("make_ref", GRID_REFS)
def test_neighbors_linear_indices(make_ref: GridRefFactory):
    device = TestOp(result_types=[IndexType()] * 2)
    op = NeighborsLinearIndicesOp(make_ref(), device.results, [1])
    assert op.grid == SymbolRefAttr("grid")
    assert op.device == device.results
    assert op.split_axes.get_values() == (1,)
    assert op.result_types == (IndexType(), IndexType())
    ModuleOp([_grid(), device, op]).verify()
