from abc import ABC


class DialectInterface(ABC):
    """
    A base class for dialects' interfaces.
    They usually define functionality which is dialect specific to some transformation.

    For example DialectInlinerInterface defines which dialect operations can be inlined and how.
    Dialects will implement this interface and the inlining transformation will query them through the base interface.

    The design logic tries to follow MLIR's dialect interfaces closely
    https://mlir.llvm.org/docs/Interfaces/#dialect-interfaces
    """
