from .ir import Operation
from .builder import Builder

# set up the operation callback for implicit construction


def _op_init_callback(op: Operation):
    if (b := Builder.get_implicit_builder()) is not None:
        b.insert(op)


Operation._op_init_callback = _op_init_callback  # pyright: ignore[reportPrivateUsage]
