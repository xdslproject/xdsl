from xdsl.ir import *
from xdsl.util import OpOrBlockArg, get_ssa_value, new_op


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

    def load(self, value: OpOrBlockArg, i: OpOrBlockArg,
             j: OpOrBlockArg) -> Operation:
        return self.ctx.get_op("affine.load").create(
            [get_ssa_value(value),
             get_ssa_value(i),
             get_ssa_value(j)], [self.ctx.get_attr("f32")()], {})

    def store(self, value: OpOrBlockArg, place: OpOrBlockArg, i: OpOrBlockArg,
              j: OpOrBlockArg) -> Operation:
        return self.ctx.get_op("affine.store").create([
            get_ssa_value(value),
            get_ssa_value(place),
            get_ssa_value(i),
            get_ssa_value(j)
        ], [], {})
