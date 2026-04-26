# TID 251 enforces to not import from nested irdl
# We need to skip it here to allow importing from here instead.
# If adding banned imports, add them also to pyproject.toml.
from .attributes import *  # noqa: TID251
from .constraints import *  # noqa: TID251
from .operations import *  # noqa: TID251
