"""
We use pyclip to copy text to the local pasteboard, but pyclip does not have type
annotations. We add our own mirror of the copy function here.
"""

from pyclip import copy  # pyright: ignore


def pyclip_copy(text: str) -> None:
    copy(text)
