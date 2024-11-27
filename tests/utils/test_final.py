import pytest

from xdsl.utils.runtime_final import is_runtime_final, runtime_final


class NotFinal:
    """A non-Final class."""


@runtime_final
class Final:
    """A Final class."""


def test_is_runtime_final():
    """Check that `is_runtime_final` returns the correct value."""
    assert not is_runtime_final(NotFinal)
    assert is_runtime_final(Final)


def test_final_inheritance_error():
    """Check that final classes cannot be subclassed."""
    with pytest.raises(TypeError, match="Subclassing final classes is restricted"):

        class SubFinal(Final):
            pass

        SubFinal()
