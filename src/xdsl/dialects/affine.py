from typing import Union

from xdsl.ir import *
from xdsl.util import new_op


@dataclass
class Affine:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(new_op("affine.for", 0, 0, 1))
        self.ctx.register_op(new_op("affine.load", 1, 3, 0))
        self.ctx.register_op(new_op("affine.store", 0, 4, 0))

    def for_(self, lower_bound: int, upper_bound: int,
             block: Block) -> Operation:
        op = self.ctx.get_op("affine.for").create([], [],
                                                  regions=[Region([block])])
        return op

    def load(self, value: Union[Operation, SSAValue],
             i: Union[Operation, SSAValue], j: Union[Operation,
                                                     SSAValue]) -> Operation:
        return self.ctx.get_op("affine.load").create(
            [SSAValue.get(value),
             SSAValue.get(i),
             SSAValue.get(j)], [self.ctx.get_attr("f32")()], {})

    def store(self, value: Union[Operation, SSAValue],
              place: Union[Operation, SSAValue], i: Union[Operation, SSAValue],
              j: Union[Operation, SSAValue]) -> Operation:
        return self.ctx.get_op("affine.store").create([
            SSAValue.get(value),
            SSAValue.get(place),
            SSAValue.get(i),
            SSAValue.get(j)
        ], [], {})
