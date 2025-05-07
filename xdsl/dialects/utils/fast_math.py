from abc import ABC
from dataclasses import dataclass

from xdsl.ir import BitEnumAttribute
from xdsl.utils.str_enum import StrEnum


class FastMathFlag(StrEnum):
    """
    Values specifying fast math behaviour of an arithmetic operation.
    """

    REASSOC = "reassoc"
    NO_NANS = "nnan"
    NO_INFS = "ninf"
    NO_SIGNED_ZEROS = "nsz"
    ALLOW_RECIP = "arcp"
    ALLOW_CONTRACT = "contract"
    APPROX_FUNC = "afn"


@dataclass(frozen=True, init=False)
class FastMathAttrBase(BitEnumAttribute[FastMathFlag], ABC):
    """
    Base class for attributes defining fast math behavior of arithmetic operations.
    """

    none_value = "none"
    all_value = "fast"
