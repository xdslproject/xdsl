from typing import Sequence, Mapping

from .ir import SSAValue, Attribute, Block, Region, Operation
from .irdl import irdl_op_model_builder


class IRDLOperation(Operation):

    def __init__(
        self,
        operands: Sequence[SSAValue | Operation
                           | Sequence[SSAValue | Operation] | None]
        | None = None,
        result_types: Sequence[Attribute | Sequence[Attribute]]
        | None = None,
        attributes: Mapping[str, Attribute | None] | None = None,
        successors: Sequence[Block] | None = None,
        regions: Sequence[Region | Sequence[Operation] | Sequence[Block]
                          | Sequence[Region | Sequence[Operation]
                                     | Sequence[Block]]]
        | None = None):

        op_def = type(self).irdl_definition
        assert op_def is not None

        model = irdl_op_model_builder(op_def, operands, result_types,
                                      attributes, successors, regions)
        super().__init__(**vars(model))
