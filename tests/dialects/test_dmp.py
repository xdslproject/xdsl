from xdsl.dialects.experimental import dmp
from xdsl.dialects.experimental.dmp import ExchangeDeclarationAttr, ShapeAttr


def flat_face_exchanges(
    shape: ShapeAttr, dim: int
) -> tuple[ExchangeDeclarationAttr, ...]:
    # we need access to the _flat_face_exchanges_for_dim method in order to test it
    # since this is a private function, and pyright will yell whenever it's accessed,
    # we have this wrapper function here that takes care of making the private publicly
    # accessible in the context of this test.
    func = dmp._flat_face_exchanges_for_dim  # pyright: ignore[reportPrivateUsage]
    return func(shape, dim)


def test_decomp_flat_face_3d():
    shape = ShapeAttr.from_index_attrs(
        (0, 0, 0),  # buff lb
        (4, 4, 4),  # core lb
        (14, 14, 14),  # core ub
        (18, 18, 18),  # buff ub
    )

    ex_pos_x, ex_neg_x = flat_face_exchanges(shape, 0)

    assert ex_pos_x.offset == (14, 4, 4)
    assert ex_pos_x.size == (4, 10, 10)
    assert ex_pos_x.source_offset == (-4, 0, 0)
    assert ex_pos_x.neighbor == (1, 0, 0)

    assert ex_neg_x.offset == (0, 4, 4)
    assert ex_neg_x.size == (4, 10, 10)
    assert ex_neg_x.source_offset == (4, 0, 0)
    assert ex_neg_x.neighbor == (-1, 0, 0)

    ex_pos_y, ex_neg_y = flat_face_exchanges(shape, 1)

    assert ex_pos_y.offset == (4, 14, 4)
    assert ex_pos_y.size == (10, 4, 10)
    assert ex_pos_y.source_offset == (0, -4, 0)
    assert ex_pos_y.neighbor == (0, 1, 0)

    assert ex_neg_y.offset == (4, 0, 4)
    assert ex_neg_y.size == (10, 4, 10)
    assert ex_neg_y.source_offset == (0, 4, 0)
    assert ex_neg_y.neighbor == (0, -1, 0)

    ex_pos_z, ex_neg_z = flat_face_exchanges(shape, 2)

    assert ex_pos_z.offset == (4, 4, 14)
    assert ex_pos_z.size == (10, 10, 4)
    assert ex_pos_z.source_offset == (0, 0, -4)
    assert ex_pos_z.neighbor == (0, 0, 1)

    assert ex_neg_z.offset == (4, 4, 0)
    assert ex_neg_z.size == (10, 10, 4)
    assert ex_neg_z.source_offset == (0, 0, 4)
    assert ex_neg_z.neighbor == (0, 0, -1)


def test_decomp_flat_face_4d():
    shape = ShapeAttr.from_index_attrs(
        (0, 0, 0, 0),  # buff lb
        (4, 4, 4, 4),  # core lb
        (14, 14, 14, 14),  # core ub
        (18, 18, 18, 18),  # buff ub
    )

    ex_pos_y, ex_neg_y = flat_face_exchanges(shape, 1)

    assert ex_pos_y.offset == (4, 14, 4, 4)
    assert ex_pos_y.size == (10, 4, 10, 10)
    assert ex_pos_y.source_offset == (0, -4, 0, 0)
    assert ex_pos_y.neighbor == (0, 1, 0, 0)

    assert ex_neg_y.offset == (4, 0, 4, 4)
    assert ex_neg_y.size == (10, 4, 10, 10)
    assert ex_neg_y.source_offset == (0, 4, 0, 0)
    assert ex_neg_y.neighbor == (0, -1, 0, 0)


def test_decomp_with_overflow():
    shape = ShapeAttr.from_index_attrs(
        (0, 0, 0),  # buff lb
        (2, 2, 2),  # core lb
        (3, 3, 102),  # core ub
        (5, 5, 104),  # buff ub
    )

    exchanges = tuple(flat_face_exchanges(shape, 0))
    assert len(exchanges) == 4

    ex_px_1, ex_px_2, ex_nx_1, ex_nx_2 = exchanges

    # all exchanges are of size (1, 1, 100)
    assert all(ex.size == (1, 1, 100) for ex in exchanges)

    # the first exchange is closer to the core region
    assert ex_px_1.offset == (3, 2, 2)
    # and has a source offset of (-1, 0, 0)
    assert ex_px_1.source_offset == (-1, 0, 0)
    # the second exchange is farther away
    assert ex_px_2.offset == (4, 2, 2)
    # and has a source offset of twice that
    assert ex_px_2.source_offset == (-2, 0, 0)

    # same for negative x, first exchange is closer to the core
    assert ex_nx_1.offset == (1, 2, 2)
    # and has a source offset of (1, 0, 0)
    assert ex_nx_1.source_offset == (1, 0, 0)
    # second is farther away
    assert ex_nx_2.offset == (0, 2, 2)
    # and has a source offset of (2, 0, 0)
    assert ex_nx_2.source_offset == (2, 0, 0)

    assert ex_px_1.neighbor == (1, 0, 0)
    assert ex_px_2.neighbor == (2, 0, 0)

    assert ex_nx_1.neighbor == (-1, 0, 0)
    assert ex_nx_2.neighbor == (-2, 0, 0)
