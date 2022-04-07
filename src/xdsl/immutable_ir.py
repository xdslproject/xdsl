from __future__ import annotations
from optparse import Option
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *


@dataclass
class ImmutableSSAValue:
    typ: Attribute


@dataclass
class ImmutableOpResultView(ImmutableSSAValue):
    op: ImmutableOperation
    result_index: int


@dataclass
class ImmutableBlockArgument(ImmutableSSAValue):
    block: ImmutableBlock
    index: int


@dataclass
class ImmutableRegion:
    blocks: FrozenList[ImmutableBlock]

    @property
    def block(self):
        return self.blocks[0]

    @staticmethod
    def from_immutable_operation_list(
            ops: List[ImmutableOperation]) -> ImmutableRegion:
        block = ImmutableBlock.from_immutable_ops(ops)
        return ImmutableRegion([block])

    @staticmethod
    def from_operation_list(ops: List[Operation]) -> ImmutableRegion:
        block = ImmutableBlock.from_ops(ops)
        return ImmutableRegion([block])

    @staticmethod
    def from_immutable_block_list(
            blocks: List[ImmutableBlock]) -> ImmutableRegion:
        return ImmutableRegion(blocks)

    @staticmethod
    def from_block_list(blocks: List[Block]) -> ImmutableRegion:
        immutable_blocks = [
            ImmutableBlock.from_block(block) for block in blocks
        ]
        return ImmutableRegion(immutable_blocks)

    def walk(self, fun: Callable[[Operation], None]) -> None:
        for block in self.blocks:
            block.walk(fun)


@dataclass
class ImmutableBlock:
    args: FrozenList[ImmutableBlockArgument]
    ops: FrozenList[ImmutableOperation]
    # parent: Optional[ImmutableRegion] = field(default=None, init=False)

    @staticmethod
    def from_block(block: Block) -> ImmutableBlock:
        context: dict[Operation, ImmutableOperation] = {}
        immutableOps = [
            ImmutableOperation.from_op(op, context) for op in block.ops
        ]
        for key in context:
            print(key.name + ", " + context[key].name)
        args = [
            ImmutableBlockArgument(arg.typ, None, arg.index)
            for arg in block.args
        ]
        newBlock = ImmutableBlock(block._args, immutableOps)
        for arg in args:
            arg.block = newBlock
        return newBlock

    @staticmethod
    def from_immutable_ops(ops: List[ImmutableOperation]) -> ImmutableBlock:
        return ImmutableBlock([], ops)

    @staticmethod
    def from__ops(ops: List[Operation]) -> ImmutableBlock:
        context: dict[Operation, ImmutableOperation] = {}
        immutable_ops = [ImmutableOperation.from_op(op, context) for op in ops]
        return ImmutableBlock.from_immutable_ops(immutable_ops)

    def walk(self, fun: Callable[[Operation], None]) -> None:
        for op in self.ops:
            op.walk(fun)


@dataclass
class ImmutableOperation:
    name: str
    _op: Operation
    operands: FrozenList[ImmutableOpResultView]  # could also be BlockArg
    regions: FrozenList[ImmutableRegion]

    @property
    def region(self):
        return self.regions[0]

    @staticmethod
    def from_op(
            op: Operation,
            context: dict[Operation,
                          ImmutableOperation]) -> ImmutableOperation:
        assert isinstance(op, Operation)

        operands: List[ImmutableOpResultView] = []
        for operand in op.operands:
            assert (isinstance(operand, OpResult))
            operands.append(
                ImmutableOpResultView(operand.typ, context[operand.op],
                                      operand.result_index))

        regions: List[ImmutableRegion] = []
        for region in op.regions:
            regions.append(ImmutableRegion.from_block_list(region.blocks))

        immutableOp = ImmutableOperation("immutable." + op.name, op, operands,
                                         regions)

        context[op] = immutableOp
        return immutableOp

    def is_op(self, SomeOpClass):
        # print(self)
        if self is not None and isinstance(self._op, SomeOpClass):
            return self
        else:
            return None

    def get_attribute(self, name: str) -> Attribute:
        return self._op.attributes[name]

    def walk(self, fun: Callable[[Operation], None]) -> None:
        fun(self)
        for region in self.regions:
            region.walk(fun)