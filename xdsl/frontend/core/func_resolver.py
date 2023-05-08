from __future__ import annotations

import ast
import importlib
import inspect
from typing import Any, Callable, Dict, TypeVar

from dataclasses import dataclass
from typing import Callable, Type
from xdsl.frontend.exception import FrontendProgramException


def resolve_func_name(module_name: str, func_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    if not hasattr(module, func_name):
        raise FrontendProgramException(
            f"Internal failure: operation '{func_name}' does not exist "
            f"in module '{module_name}'."
        )
    return getattr(module, func_name)
