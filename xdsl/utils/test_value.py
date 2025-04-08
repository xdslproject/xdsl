from typing import Generic

from xdsl.ir import AttributeCovT, Block, Operation, SSAValue


class TestSSAValue(Generic[AttributeCovT], SSAValue[AttributeCovT]):
    @property
    def owner(self) -> Operation | Block:
        import pytest

        pytest.fail("Attempting to get the owner of a `TestSSAValue`")
