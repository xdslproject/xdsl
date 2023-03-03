from typing import Callable, ParamSpec, TypeVar
import warnings

_T = TypeVar('_T')
_P = ParamSpec('_P')


def deprecated(reason: str):
    """Deprecate the use of a method, and provide a warning message."""

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:

        def new_func(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            warnings.warn(
                f'Call to deprecated method {func.__name__}: {reason}')
            return func(*args, **kwargs)

        return new_func

    return decorator


def deprecated_constructor(func: Callable[_P, _T]) -> Callable[_P, _T]:
    return deprecated(f'use the constructor (`ClassName(...)`) instead.')(func)
