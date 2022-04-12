from __future__ import annotations
from optparse import Option
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *
from xdsl.rewriter import Rewriter


@dataclass
class ImmutableSSAValue:
    typ: Attribute


@dataclass
class ImmutableOpResult(ImmutableSSAValue):
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
    operands: FrozenList[SSAValue]
    results: FrozenList[ImmutableOpResult]
    regions: FrozenList[ImmutableRegion]
    parentBlock: Optional[ImmutableBlock] = None

    @property
    def region(self):
        return self.regions[0]

    def __post_init__(self):
        for result in self.results:
            result.op = self

    @staticmethod
    def from_op(
        op: Operation,
        context: dict[Operation,
                      ImmutableOperation] = None) -> ImmutableOperation:
        assert isinstance(op, Operation)
        if context is None:
            context = {}

        operands: List[ImmutableSSAValue] = []
        for operand in op.operands:
            assert (isinstance(operand, OpResult))
            # Small workaround when we do not already have an ImmutableOperation for the operands
            operands.append(
                ImmutableOpResult(
                    operand.typ, context[operand.op] if operand.op in context
                    else ImmutableOperation.from_op(operand.op),
                    operand.result_index))

        results: List[ImmutableOpResult] = []
        for result in op.results:
            results.append(
                ImmutableOpResult(result.typ, None, result.result_index))

        regions: List[ImmutableRegion] = []
        for region in op.regions:
            regions.append(ImmutableRegion.from_block_list(region.blocks))

        immutableOp = ImmutableOperation("immutable." + op.name, op, operands,
                                         results, regions)

        context[op] = immutableOp
        return immutableOp

    def create_new(OpType: OperationType,
                   immutable_operands: List[ImmutableOpResult],
                   op: Operation) -> List[ImmutableOperation]:
        # TODO: finish this
        # this should also clone all operations in the operands, which are not newly created
        # that are those who already are attached to a parentBlock
        dependantOperations = []

        newOp = OpType.create(op.operands,
                              [result.typ for result in op.results],
                              op.attributes.copy(), op.successors.copy())

        # This should actually return a list of operations:
        #   the new op itself and all cloned dependant ops
        return [ImmutableOperation.from_op(newOp)]

    #         @classmethod
    # def create(cls: typing.Type[OperationType],
    #            operands: Optional[List[SSAValue]] = None,
    #            result_types: Optional[List[Attribute]] = None,
    #            attributes: Optional[Dict[str, Attribute]] = None,
    #            successors: Optional[List[Block]] = None,
    #            regions: Optional[List[Region]] = None) -> OperationType:
    #     return Operation.with_result_types(cls, operands, result_types,
    #                                        attributes, successors, regions)

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