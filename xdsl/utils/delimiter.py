from enum import Enum


class Delimiter(Enum):
    """
    Supported delimiters for parsing and printing.
    """

    PAREN = ("(", ")")
    ANGLE = ("<", ">")
    SQUARE = ("[", "]")
    BRACES = ("{", "}")
    METADATA_TOKEN = ("{-#", "#-}")
    NONE = None
