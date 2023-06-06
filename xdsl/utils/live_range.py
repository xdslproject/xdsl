from dataclasses import dataclass
from xdsl.ir import Block, OpResult, SSAValue
from xdsl.utils.exceptions import InvalidIRException


@dataclass
class LiveRange:
    value: SSAValue
    start: int
    end: int

    def __init__(self, value: SSAValue) -> None:
        if not value.owner:
            raise InvalidIRException(
                "Cannot calculate live range for value not belonging to a block"
            )
        if isinstance(value.owner, Block):
            raise NotImplementedError("Not support block arguments yet")

        owner = value.owner

        if isinstance(value, OpResult) and isinstance(owner.parent, Block):
            self.value = value
            self.start = owner.parent.get_operation_index(value.owner)
            self.end = self.start

            for use in value.uses:
                if parent_block := use.operation.parent:
                    self.end = max(
                        self.end,
                        parent_block.get_operation_index(use.operation),
                    )
                else:
                    raise NotImplementedError(
                        "Cannot calculate live range for value across blocks"
                    )
