import warnings

# TID 251 enforces to not import from core
# We need to skip it here to allow importing from here instead.
# If adding banned imports, add them also to pyproject.toml.
from .core import *  # noqa: TID251


def __getattr__(name: str):
    if name in ("EnumAttribute", "EnumType"):
        from xdsl.dialects.utils.enum_attribute import (  # noqa: TID251
            EnumAttribute,
            EnumType,
        )

        warnings.warn(
            f"Importing '{name}' from 'xdsl.ir' is deprecated. "
            f"Please use 'from xdsl.dialects.utils import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return EnumAttribute if name == "EnumAttribute" else EnumType
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
