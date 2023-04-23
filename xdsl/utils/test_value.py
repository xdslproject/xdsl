from xdsl.ir import Block, Operation, SSAValue


class TestSSAValue(SSAValue):
    @property
    def owner(self) -> Operation | Block:
        assert False, "Attempting to get the owner of a `TestSSAValue`"
