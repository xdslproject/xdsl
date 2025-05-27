from collections.abc import Sequence

import pytest

from xdsl.ir import Attribute
from xdsl.irdl import ConstraintContext, RangeConstraint


class AnyRangeConstraint(RangeConstraint):
    """Constraint for testing default infer"""

    def verify(
        self, attrs: Sequence[Attribute], constraint_context: ConstraintContext
    ) -> None:
        return

    def verify_length(self, length: int, constraint_context: ConstraintContext) -> None:
        return


def test_failing_inference():
    with pytest.raises(
        ValueError, match="Cannot infer range from constraint AnyRangeConstraint()"
    ):
        AnyRangeConstraint().infer(ConstraintContext(), length=None)
