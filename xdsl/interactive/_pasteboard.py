"""
We use pyclip to copy text to the local pasteboard, but pyclip does not have type
annotations. We add our own mirror of the copy function here.
"""

from collections.abc import Callable

_test_pyclip_callback: Callable[[str], None] | None = None
"""Used in tests."""


def pyclip_copy(text: str) -> None:
    global _test_pyclip_callback
    if _test_pyclip_callback is not None:
        _test_pyclip_callback(text)
        return

    import pyclip  # pyright: ignore[reportMissingTypeStubs]

    pyclip.copy(text)  # pyright: ignore[reportUnknownMemberType]
