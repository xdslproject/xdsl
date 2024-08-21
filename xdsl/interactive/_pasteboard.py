"""
We use pyclip to copy text to the local pasteboard, but pyclip does not have type
annotations. We add our own mirror of the copy function here.
"""

import pyclip  # pyright: ignore[reportMissingTypeStubs]


def pyclip_copy(text: str) -> None:
    pyclip.copy(text)  # pyright: ignore[reportUnknownMemberType]
