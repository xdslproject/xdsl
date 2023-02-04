from __future__ import annotations

import xdsl.dialects.builtin as builtin

from typing import Any, Callable, Generic, TypeAlias, TypeVar, Literal
from xdsl.dialects.builtin import _FrontendType

Type = TypeVar("Type", covariant=True)


class Secret(Generic[Type], _FrontendType):
    """
    Represents a Secret type in the frontend.
    """
