from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import IO, ClassVar

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.arg_spec import ArgSpecConvertible


@dataclass(frozen=True)
class Target(ArgSpecConvertible):
    """
    A Target is a named output backend that serializes a ModuleOp to a stream.

    Targets can accept arguments following the same ``name{arg=val ...}`` syntax
    used by passes.  Subclasses must be decorated with
    ``@dataclass(frozen=True)`` and provide a ``name`` class variable.
    """

    name: ClassVar[str]

    @abstractmethod
    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None: ...
