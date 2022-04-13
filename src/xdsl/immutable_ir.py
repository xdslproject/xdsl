from __future__ import annotations
from optparse import Option
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *
from xdsl.rewriter import Rewriter


@dataclass
class ImmutableSSAValue:
    typ: Attribute

    def get_op(self) -> ImmutableOperation:
        if isinstance(self, ImmutableOpResult):
            return self.op
        return None  # type: ignore

    def get_mutable(self) -> Optional[OpResult]:
        if isinstance(self, ImmutableOpResult):
            return self.get_mutable_operand()
        elif isinstance(self, ImmutableBlockArgument):
            # TODO: do
            raise NotImplemented()
        else:
            return None


@dataclass
class ImmutableOpResult(ImmutableSSAValue):
    op: ImmutableOperation  # for initialization purposes
    result_index: int

    def get_mutable_operand(self) -> OpResult:
        assert self.op is not None
        return self.op._op.results[self.result_index]


@dataclass
class ImmutableBlockArgument(ImmutableSSAValue):
    block: ImmutableBlock  # for initialization purposes
    index: int


@dataclass
class ImmutableRegion:
    blocks: FrozenList[ImmutableBlock]
    parent_op: Optional[ImmutableOperation] = None

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

    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
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
        # TODO: operations using block arguments are not handled properly
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

    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
        for op in self.ops:
            op.walk(fun)


def get_immutable_copy(op: Operation) -> ImmutableOperation:
    return ImmutableOperation.from_op(op, {})


@dataclass
class ImmutableOperation:
    name: str
    _op: Operation
    operands: FrozenList[ImmutableSSAValue]
    results: FrozenList[ImmutableOpResult]
    successors: FrozenList[ImmutableBlock]
    regions: FrozenList[ImmutableRegion]
    parentBlock: Optional[ImmutableBlock] = None

    @property
    def region(self):
        return self.regions[0]

    @property
    def result_types(self) -> List[Attribute]:
        return [result.typ for result in self.results]

    def __post_init__(self):
        for result in self.results:
            result.op = self

    @staticmethod
    def from_op(
        op: Operation,
        operationMap: Optional[Dict[Operation, ImmutableOperation]] = None,
        blockMap: Optional[Dict[Block, ImmutableBlock]] = None
    ) -> ImmutableOperation:
        assert isinstance(op, Operation)
        if operationMap is None:
            operationMap = {}
        if blockMap is None:
            blockMap = {}

        operands: List[ImmutableSSAValue] = []
        for operand in op.operands:
            assert (isinstance(operand, OpResult))
            # Small workaround when we do not already have an ImmutableOperation for the operands
            operands.append(
                ImmutableOpResult(
                    operand.typ,
                    operationMap[operand.op] if operand.op in operationMap else
                    ImmutableOperation.from_op(operand.op),
                    operand.result_index))

        results: List[ImmutableOpResult] = []
        for result in op.results:
            results.append(
                ImmutableOpResult(result.typ, None,
                                  result.result_index))  # type: ignore

        successors: List[ImmutableBlock] = []
        for successor in op.successors:
            if successor in blockMap:
                successors.append(blockMap[successor])
            else:
                newImmutableSuccessor = ImmutableBlock.from_block(successor)
                blockMap[successor] = newImmutableSuccessor
                successors.append(newImmutableSuccessor)

        regions: List[ImmutableRegion] = []
        for region in op.regions:
            regions.append(ImmutableRegion.from_block_list(region.blocks))

        immutableOp = ImmutableOperation("immutable." + op.name, op, operands,
                                         results, successors, regions)

        for region in immutableOp.regions:
            region.parent_op = immutableOp

        operationMap[op] = immutableOp
        return immutableOp

    @classmethod
    def create_new(
        cls,
        op_type: OperationType,
        immutable_operands: Optional[List[ImmutableSSAValue]] = None,
        result_types: Optional[List[Attribute]] = None,
        attributes: Optional[Dict[str, Attribute]] = None,
        successors: Optional[List[ImmutableBlock]] = None,
        regions: Optional[List[ImmutableRegion]] = None
    ) -> List[ImmutableOperation]:
        """Creates new mutable operations and returns an immutable view on them."""

        if immutable_operands is None:
            immutable_operands = []
        if result_types is None:
            result_types = []
        if attributes is None:
            attributes = {}  # = original_mutable_op.attributes.copy()
        if successors is None:
            successors = []  # original_mutable_op.successors
        if regions is None:
            regions = []

        dependantOperations = []
        operands = []
        for imm_operand in immutable_operands:
            if isinstance(imm_operand,
                          ImmutableOpResult) and (op := imm_operand.get_op(
                          )) is not None and op.parentBlock is not None:
                # parent block set means we have to clone the op
                clonedImmutableOps = op.create_new(
                    op._op.__class__,
                    immutable_operands=imm_operand.get_op().operands,
                    result_types=[result.typ for result in op._op.results],
                    attributes=op._op.attributes.copy())
                dependantOperations.extend(clonedImmutableOps)
                operands.append(clonedImmutableOps[-1].results[
                    imm_operand.result_index].get_mutable())
            else:
                operands.append(imm_operand.get_mutable())

        # TODO: get Regions from the ImmutableRegions
        # mutable_regions = []
        # for region in regions:
        #     mutable_blocks = []
        #     for block in region.blocks:

        # maybe it is easier

        # This will not work properly for blocks where the arguments are used by ops inside
        # mutable_blocks.append(Block.from_arg_types([arg.typ for arg in block.args]))

        # successors is ImmutableBlock, not Block here!
        newOp = op_type.create(operands=list(operands),
                               result_types=result_types,
                               attributes=attributes,
                               successors=successors,
                               regions=regions)

        return dependantOperations + [ImmutableOperation.from_op(newOp)]

    def get_attribute(self, name: str) -> Attribute:
        return self._op.attributes[name]

    def get_attributes_copy(self) -> Dict[str, Attribute]:
        return self._op.attributes.copy()

    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
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


def isa(op: Optional[ImmutableOperation], SomeOpClass):
    if op is not None and isinstance(op._op, SomeOpClass):
        return True
    else:
        return False