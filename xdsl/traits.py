from xdsl.ir import OpTrait


class Pure(OpTrait):
    """A trait that signals that an operation has no side effects."""
