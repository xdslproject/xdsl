from xdsl.dialects.experimental import dmp
from xdsl.dialects.experimental.dmp import ExchangeDeclarationAttr, ShapeAttr


def flat_face_exchanges(
    shape: ShapeAttr, dim: int
) -> tuple[ExchangeDeclarationAttr, ExchangeDeclarationAttr]:
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
