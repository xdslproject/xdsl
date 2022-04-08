from __future__ import annotations
from optparse import Option
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *
from xdsl.rewriter import Rewriter


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

    def __post_init__(self):
        for op in self.ops:
            op.parentBlock = self

    @staticmethod
    def from_block(block: Block) -> ImmutableBlock:
        context: dict[Operation, ImmutableOperation] = {}
        immutableOps = [
            ImmutableOperation.from_op(op, context) for op in block.ops
        ]
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
    parentBlock: Optional[ImmutableBlock] = None

    @property
    def region(self):
        return self.regions[0]

    @staticmethod
    def from_op(
        op: Operation,
        context: dict[Operation,
                      ImmutableOperation] = None) -> ImmutableOperation:
        assert isinstance(op, Operation)
        if context is None:
            context = {}

        operands: List[ImmutableOpResultView] = []
        for operand in op.operands:
            assert (isinstance(operand, OpResult))
            # Small workaround when we do not already have an ImmutableOperation for the operands
            operands.append(
                ImmutableOpResultView(
                    operand.typ, context[operand.op] if operand.op in context
                    else ImmutableOperation.from_op(operand.op),
                    operand.result_index))

        regions: List[ImmutableRegion] = []
        for region in op.regions:
            regions.append(ImmutableRegion.from_block_list(region.blocks))

        immutableOp = ImmutableOperation("immutable." + op.name, op, operands,
                                         regions)

        context[op] = immutableOp
        return immutableOp

    def get_attribute(self, name: str) -> Attribute:
        return self._op.attributes[name]

    def walk(self, fun: Callable[[Operation], None]) -> None:
        fun(self)
        for region in self.regions:
            region.walk(fun)

    def get_mutable_copy(self) -> Operation:
        return self._op.clone()

    def replace_with(self, ops: List[ImmutableOperation]):
        assert (isinstance(ops, List))
        assert (all([isinstance(op, ImmutableOperation) for op in ops]))
        rewriter = Rewriter()
        rewriter.replace_op(self._op, [op._op for op in ops])


def isa(op: ImmutableOperation, SomeOpClass):
    if op is not None and isinstance(op._op, SomeOpClass):
        return True
    else:
        return False