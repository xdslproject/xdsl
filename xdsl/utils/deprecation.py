import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

# We could use the `decorated` library, but it did not seem necessary for
# this simple use case.
# If we ever need more features from the library, we can switch to it.

_T = TypeVar("_T")
_P = ParamSpec("_P")


def deprecated(reason: str):
    """Deprecate the use of a method, and provide a warning message."""

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def new_func(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            warnings.warn(
                f"Call to deprecated method {str(func).split(' ')[1]}: {reason}"
            )
            return func(*args, **kwargs)

        return new_func

    return decorator


def deprecated_constructor(func: Callable[_P, _T]) -> Callable[_P, _T]:
    # TOFIX: improve printing
    return deprecated(f"{'use the constructor (`ClassName(...)`) instead.'}")(func)
