from xdsl.ir import Block, Operation, SSAValue


class TestSSAValue(SSAValue):
    @property
    def owner(self) -> Operation | Block:
        assert False, "Attempting to get the owner of a `TestSSAValue`"

    def __eq__(self, other: object) -> bool:
        return self is other

    # This might be problematic, as the superclass is not hashable ...
    def __hash__(self) -> int:  # type: ignore
        return id(self)
