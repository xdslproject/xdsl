from __future__ import annotations

from typing import Any, Callable


class FrontendType:
    """Represents any type in the frontend."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        raise NotImplementedError()


def frontend_op(frontend, operation):
    def decorator(func):
        frontend.add_mapping(func, operation)
        return func

    return decorator
