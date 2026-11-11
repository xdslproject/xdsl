from collections.abc import Callable

from xdsl.utils.frontend import Frontend


def get_all_frontends() -> dict[str, Callable[[], type[Frontend]]]:
    """Return the list of all available frontends."""

    def get_mlir():
        from xdsl.frontend.mlir import MLIRFrontend

        return MLIRFrontend

    return {
        "mlir": get_mlir,
    }
