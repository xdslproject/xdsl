from dataclasses import dataclass
from xdsl.ir import Block, OpResult, SSAValue
from xdsl.utils.exceptions import InvalidIRException


@dataclass(init=False, frozen=True)
class LiveRange:
    """
    Represents the live range of an SSA value as half-open interval using the
    index of an Operation within its parent Block.
    """

    value: SSAValue
    start: int
    end: int

    def __init__(
        self, value: SSAValue, start: int | None = None, end: int | None = None
    ) -> None:
        if start is not None and end is not None:
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "start", start)
            object.__setattr__(self, "end", end)
        else:
            if not value.owner:
                raise InvalidIRException(
                    "Cannot calculate live range for value not belonging to a block"
                )
            if isinstance(value.owner, Block):
                raise NotImplementedError("Cannot support block arguments")

            owner = value.owner

            if isinstance(value, OpResult) and isinstance(owner.parent, Block):
                object.__setattr__(self, "value", value)
                object.__setattr__(
                    self, "start", owner.parent.get_operation_index(value.owner)
                )

                end = self.start
                for use in value.uses:
                    if parent_block := use.operation.parent:
                        end = max(
                            end,
                            parent_block.get_operation_index(use.operation),
                        )
                    else:
                        raise NotImplementedError(
                            "Cannot calculate live range for value across blocks"
                        )

                object.__setattr__(self, "end", end)
