import warnings

from xdsl.ir import Dialect

from . import abstract_ops, attrs, ops

_DEPRECATED_SUBMODULES = (attrs, abstract_ops, ops)


def __getattr__(name: str):
    # Check if the requested attribute exists in the new submodules.
    for module in _DEPRECATED_SUBMODULES:
        if not hasattr(module, name):
            continue
        warnings.warn(
            f"Importing '{name}' directly from 'xdsl.dialects.linalg' is deprecated. "
            f"Please use 'from xdsl.dialects.linalg.{module.__name__.split('.')[-1]} import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(module, name)

    # If it's not in any split module, raise the standard AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


Linalg = Dialect(
    "linalg",
    [
        ops.GenericOp,
        ops.YieldOp,
        ops.IndexOp,
        ops.AddOp,
        ops.ExpOp,
        ops.LogOp,
        ops.SubOp,
        ops.SqrtOp,
        ops.SelectOp,
        ops.FillOp,
        ops.CopyOp,
        ops.MaxOp,
        ops.MinOp,
        ops.MulOp,
        ops.TransposeOp,
        ops.MatmulOp,
        ops.QuantizedMatmulOp,
        ops.PoolingNchwMaxOp,
        ops.Conv2DNchwFchwOp,
        ops.Conv2DNhwgcGfhwcOp,
        ops.Conv2DNhwc_HwcfOp,
        ops.Conv2DNgchwGfchwOp,
        ops.Conv2DNgchwFgchwOp,
        ops.Conv2DNhwc_FhwcOp,
        ops.BroadcastOp,
        ops.ReduceOp,
    ],
    [
        attrs.IteratorTypeAttr,
    ],
)
