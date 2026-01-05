"""
We use pyclip to copy text to the local pasteboard, but pyclip does not have type
annotations. We add our own mirror of the copy function here.
"""


def pyclip_copy(text: str) -> None:
    import pyclip  # pyright: ignore[reportMissingTypeStubs]

    pyclip.copy(text)  # pyright: ignore[reportUnknownMemberType]
