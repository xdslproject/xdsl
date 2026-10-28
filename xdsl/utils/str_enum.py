"""
Kept for backward compatibility of the import path: ``StrEnum`` is part of the
standard library since Python 3.11, which xDSL now requires as a minimum.
"""

from enum import StrEnum

__all__ = ["StrEnum"]
