# TID 251 enforces to not import from nested irdl
# We need to skip it here to allow importing from here instead.
from .irdl import *  # noqa: TID251
