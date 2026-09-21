import io

import pytest

from xdsl.dialects import transform
from xdsl.dialects.builtin import DenseArrayBase, i1, i64
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value


def _print(op: transform.TileOp) -> str:
    out = io.StringIO()
    Printer(stream=out).print_op(op)
    return out.getvalue()


@pytest.mark.parametrize(
    "static_sizes, scalable_sizes, expected",
    [
        (None, None, "tile_sizes []"),
        ([32, 0, 8], None, "tile_sizes [32, 0, 8]"),
        ([32, 4, 8], [0, 1, 0], "tile_sizes [32, [4], 8]"),
    ],
)
def test_tile_op_prints_tile_sizes(
    static_sizes: list[int] | None,
    scalable_sizes: list[int] | None,
    expected: str,
):
    target = create_ssa_value(transform.AnyOpType())
    op = transform.TileOp(
        target, static_sizes=static_sizes, scalable_sizes=scalable_sizes
    )
    op.verify()
    assert expected in _print(op)


def test_tile_op_result_count_matches_nonzero_tile_sizes():
    target = create_ssa_value(transform.AnyOpType())
    op = transform.TileOp(target, static_sizes=[32, 0, 8])
    assert op.static_sizes == DenseArrayBase.from_list(i64, [32, 0, 8])
    assert len(op.loops) == 2


def test_tile_op_constructor_converts_sequences():
    target = create_ssa_value(transform.AnyOpType())
    op = transform.TileOp(
        target, static_sizes=[32, 4], interchange=[1, 0], scalable_sizes=[0, 1]
    )
    op.verify()
    assert op.interchange == DenseArrayBase.from_list(i64, [1, 0])
    assert op.scalable_sizes == DenseArrayBase.from_list(i1, [0, 1])
    assert "interchange = [1, 0]" in _print(op)
