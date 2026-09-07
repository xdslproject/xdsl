import warnings

# TID 251 enforces to not import from core
# We need to skip it here to allow importing from here instead.
# If adding banned imports, add them also to pyproject.toml.
from .core import *  # noqa: TID251


def __getattr__(name: str):
    if name == "EnumAttribute":
        from xdsl.dialects.utils.enum_attribute import EnumAttribute

        warnings.warn(
            "Importing 'EnumAttribute' from 'xdsl.ir' is deprecated. "
            "Please use 'from xdsl.dialects.utils import EnumAttribute' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return EnumAttribute
    if name == "StrEnum":
        from xdsl.utils.str_enum import StrEnum

        warnings.warn(
            "Importing 'StrEnum' from 'xdsl.ir' is deprecated. "
            "Please use 'from xdsl.utils.str_enum import StrEnum' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return StrEnum
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
