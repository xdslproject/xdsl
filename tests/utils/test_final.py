import pytest

from xdsl.utils.final import final, is_final


class NotFinal:
    """A non-Final class."""


@final
class Final:
    """A Final class."""


def test_is_final():
    """Check that `is_final` returns the correct value."""
    assert not is_final(NotFinal)
    assert is_final(Final)


def test_final_inheritance_error():
    """Check that final classes cannot be subclassed."""
    with pytest.raises(TypeError, match="Subclassing final classes is restricted"):

        class SubFinal(Final):
            pass

        SubFinal()
