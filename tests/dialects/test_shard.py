import pytest

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseArrayBase,
    IndexType,
    MemRefType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    f32,
    i64,
)
from xdsl.dialects.shard import (
    GridOp,
    NeighborsLinearIndicesOp,
    ProcessLinearIndexOp,
    ProcessMultiIndexOp,
    ShardingType,
    ShardShapeOp,
    UpdateHaloOp,
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


GridRef = str | StringAttr | SymbolRefAttr

GRID_REFS: list[GridRef] = ["grid", StringAttr("grid"), SymbolRefAttr("grid")]


@pytest.mark.parametrize("grid", GRID_REFS)
def test_process_linear_index(grid: GridRef):
    op = ProcessLinearIndexOp(grid)
    assert op.grid == SymbolRefAttr("grid")
    assert op.result.type == IndexType()
    ModuleOp([_grid(), op]).verify()


@pytest.mark.parametrize(
    "grid, axes, expected_axes, num_results",
    [
        ("grid", 2, (), 2),
        (StringAttr("grid"), (1,), (1,), 1),
        ("grid", (1, 0), (1, 0), 2),
        (SymbolRefAttr("grid"), (0,), (0,), 1),
    ],
)
def test_process_multi_index(
    grid: GridRef,
    axes: tuple[int, ...] | int,
    expected_axes: tuple[int, ...],
    num_results: int,
):
    op = ProcessMultiIndexOp(grid, axes)
    assert op.grid == SymbolRefAttr("grid")
    assert op.axes.get_values() == expected_axes
    assert op.result.types == (IndexType(),) * num_results
    ModuleOp([_grid(), op]).verify()


def test_process_multi_index_wrong_rank():
    op = ProcessMultiIndexOp("grid", 3)
    with pytest.raises(
        VerifyException, match=r"Unexpected number of results 3\. Expected 2\."
    ):
        ModuleOp([_grid(), op]).verify()


@pytest.mark.parametrize("grid", GRID_REFS)
def test_neighbors_linear_indices(grid: GridRef):
    device = TestOp(result_types=[IndexType()] * 2)
    op = NeighborsLinearIndicesOp(grid, device.results, [1])
    assert op.grid == SymbolRefAttr("grid")
    assert op.device == device.results
    assert op.split_axes.get_values() == (1,)
    assert op.result_types == (IndexType(), IndexType())
    ModuleOp([_grid(), device, op]).verify()


def test_shard_shape():
    sharding = TestOp(result_types=[ShardingType()])
    index = TestOp(result_types=[IndexType()])
    op = ShardShapeOp([8, index.results[0]], sharding, [index.results[0], 1])
    assert op.dims.get_values() == (8, DYNAMIC_INDEX)
    assert op.dims_dynamic == index.results
    assert op.sharding == sharding.results[0]
    assert op.device.get_values() == (DYNAMIC_INDEX, 1)
    assert op.device_dynamic == index.results
    assert op.result.types == (IndexType(),) * 2
    op.verify()


@pytest.mark.parametrize("grid", GRID_REFS)
def test_update_halo(grid: GridRef):
    destination = TestOp(result_types=[MemRefType(f32, [2, 2])])
    size = TestOp(result_types=[i64])
    op = UpdateHaloOp(destination, grid, [[0], [1]], [1, size.results[0], 1, 1])
    assert op.grid == SymbolRefAttr("grid")
    assert op.destination == destination.results[0]
    assert [axes.get_values() for axes in op.split_axes.axes] == [(0,), (1,)]
    assert op.static_halo_sizes.get_values() == (1, DYNAMIC_INDEX, 1, 1)
    assert op.halo_sizes == size.results
    assert op.result.type == destination.result_types[0]
    ModuleOp([_grid(), destination, size, op]).verify()
