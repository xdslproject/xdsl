from xdsl.utils.hasher import Hasher


def test_hasher():
    h = Hasher()
    h.combine(1)
    hash1 = h.hash
    h.combine(2)

    j = Hasher(seed=hash1)

    assert h.hash != j.hash

    j.combine(2)

    assert h.hash == j.hash
