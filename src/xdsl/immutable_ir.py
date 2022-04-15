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

    @abstractmethod
    def get_mutable(self) -> SSAValue:
        ...


@dataclass
class ImmutableOpResult(ImmutableSSAValue):
    op: ImmutableOperation  # for initialization purposes
    result_index: int

    def get_mutable(self) -> OpResult:
        assert self.op is not None
        return self.op._op.results[self.result_index]


@dataclass
class ImmutableBlockArgument(ImmutableSSAValue):
    block: ImmutableBlock  # for initialization purposes
    index: int

    def get_mutable(self) -> BlockArgument:
        assert self.block is not None
        assert self.block._block is not None
        return self.block._block.args[self.index]


@dataclass
class ImmutableRegion:
    _region: Region
    blocks: FrozenList[ImmutableBlock]
    parent_op: Optional[ImmutableOperation] = None

    def __post_init__(self):
        for block in self.blocks:
            block.parent_region = self
        self.blocks.freeze()

    @property
    def block(self):
        return self.blocks[0]

    @classmethod
    def _create_new(
            cls,
            arg: ImmutableRegion | List[ImmutableBlock]
        | List[ImmutableOperation],
            value_map: Optional[Dict] = None,
            block_map: Optional[Dict] = None
    ) -> Tuple[ImmutableRegion, Region]:
        """Creates a new mutable region and returns an immutable view on it and the region."""
        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        match arg:
            case ImmutableRegion():
                blocks = list(arg.blocks)
            case [*blocks_] if all(
                isinstance(block, ImmutableBlock) for block in blocks_):
                blocks = blocks_
            case [*ops_
                  ] if all(isinstance(op, ImmutableOperation) for op in ops_):
                raise Exception(
                    "Creating an ImmutableRegion from ops directly is not yet implemented"
                )
                blocks = []
            case _:
                raise Exception(
                    "unsupported argument to create ImmutableRegion from.")

        mutable_blocks = []
        immutable_blocks = []
        for immutable_block in blocks:
            if immutable_block.parent_region is not None:  # type: ignore
                # if the immutable_block already has a parent_region we have to recreate it
                new_block = ImmutableBlock._create_new(
                    immutable_block, value_map, block_map)  # type: ignore
                immutable_blocks.append(new_block[0])
                mutable_blocks.append(new_block[1])
            else:
                immutable_blocks.append(immutable_block)
                mutable_blocks.append(immutable_block._block)  # type: ignore

        new_region = Region.get(mutable_blocks)
        return ImmutableRegion.from_block_list(new_region.blocks), new_region

    @classmethod
    def create_new(
        cls,
        arg: ImmutableRegion | List[ImmutableBlock] | List[ImmutableOperation]
    ) -> ImmutableRegion:
        """Creates a new mutable region and returns an immutable view on it."""
        return cls._create_new(arg)[0]

    # @staticmethod
    # def from_immutable_operation_list(
    #         ops: List[ImmutableOperation]) -> ImmutableRegion:
    #     block = ImmutableBlock.from_immutable_ops(ops)
    #     return ImmutableRegion(FrozenList([block]))

    # @staticmethod
    # def from_operation_list(ops: List[Operation]) -> ImmutableRegion:
    #     block = ImmutableBlock.from_ops(ops)
    #     return ImmutableRegion(FrozenList([block]))

    # @staticmethod
    # def from_immutable_block_list(
    #         blocks: List[ImmutableBlock]) -> ImmutableRegion:
    #     return ImmutableRegion(FrozenList(blocks))

    @staticmethod
    def from_block_list(blocks: List[Block]) -> ImmutableRegion:
        immutable_blocks = [
            ImmutableBlock.from_block(block) for block in blocks
        ]
        assert (blocks[0].parent is not None)
        return ImmutableRegion(blocks[0].parent, FrozenList(immutable_blocks))

    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
        for block in self.blocks:
            block.walk(fun)


@dataclass
class ImmutableBlock:
    _block: Block
    args: FrozenList[ImmutableBlockArgument]
    ops: FrozenList[ImmutableOperation]
    parent_region: Optional[ImmutableRegion] = None

    def __post_init__(self):
        for op in self.ops:
            op.parent_block = self
        for arg in self.args:
            arg.block = self

        self.args.freeze()
        self.ops.freeze()

    @classmethod
    def _create_new(
            cls,
            arg: ImmutableBlock | List[ImmutableOperation],
            old_block: Optional[ImmutableBlock] = None,
            value_map: Optional[Dict] = None,
            block_map: Optional[Dict] = None) -> Tuple[ImmutableBlock, Block]:
        """Creates a new mutable block and returns an immutable view on it and the mutable block itself."""
        print("create new Block")

        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        args = old_block.args if old_block is not None else []

        match arg:
            case ImmutableBlock():
                if old_block is None:
                    args.extend(list(arg.args))
                ops = list(arg.ops)
                new_block = Block.from_arg_types(
                    [block_arg.typ for block_arg in args])

                for idx, old_block_arg in enumerate(arg.args):
                    value_map[arg._block.args[idx]] = new_block.args[idx]
                block_map[arg._block] = new_block
            case [*operations] if all(
                isinstance(op, ImmutableOperation) for op in operations):
                ops = operations
                new_block = Block.from_arg_types(
                    [block_arg.typ for block_arg in args])
            case _:
                raise Exception(
                    "unsupported argument to create ImmutableBlock from.")

        for idx, old_block_arg in enumerate(args):
            value_map[old_block_arg] = new_block.args[idx]

        immutable_ops = []
        if len(ops) == 0:
            return ImmutableBlock.from_block(new_block), new_block

        if (immutable_op := ops[-1]).parent_block is not None:
            # if the immutable_op already has a parent_block we have to recreate it
            new_ops = ImmutableOperation._create_new(
                immutable_op._op.__class__, list(immutable_op.operands),
                immutable_op.result_types, immutable_op.get_attributes_copy(),
                list(immutable_op.successors), list(immutable_op.regions),
                value_map, block_map)

            immutable_ops.extend(new_ops[0])
            new_block.add_ops(new_ops[1])
        else:
            # TODO: problem if new operations are mixed with old operations?
            # I am not sure whether this even comes up? It shouldn't I think
            immutable_ops.extend(ops)
            new_block.add_ops([imm_op._op for imm_op in ops])

        # This rebuilds the ImmutableOperations we already have, but that is required currently:
        # The ImmutableOperations might need updated references to BlockArgs.
        return ImmutableBlock.from_block(new_block), new_block

    @classmethod
    def create_new(
            cls,
            arg: Union[ImmutableBlock, List[ImmutableOperation]],
            old_block: Optional[ImmutableBlock] = None) -> ImmutableBlock:
        """Creates a new mutable block and returns an immutable view on it."""
        return cls._create_new(arg, old_block)[0]

    @staticmethod
    def from_block(block: Block) -> ImmutableBlock:
        value_map: dict[SSAValue, ImmutableSSAValue] = {}
        block_map: dict[Block, ImmutableBlock] = {}

        args: List[ImmutableBlockArgument] = []
        for arg in block.args:
            immutable_arg = ImmutableBlockArgument(arg.typ, None, arg.index)
            args.append(immutable_arg)
            value_map[arg] = immutable_arg

        immutable_ops = [
            ImmutableOperation.from_op(op,
                                       value_map=value_map,
                                       block_map=block_map) for op in block.ops
        ]

        return ImmutableBlock(block, FrozenList(args),
                              FrozenList(immutable_ops))

    @staticmethod
    def _from_args(block: Block, block_args: List[ImmutableBlockArgument],
                   imm_ops: List[ImmutableOperation]) -> ImmutableBlock:
        return ImmutableBlock(block, FrozenList(block_args),
                              FrozenList(imm_ops))

    # @staticmethod
    # def from_immutable_ops(ops: List[ImmutableOperation]) -> ImmutableBlock:
    #     return ImmutableBlock(FrozenList([]), FrozenList(ops))

    # @staticmethod
    # def from_ops(ops: List[Operation]) -> ImmutableBlock:
    #     context: dict[Operation, ImmutableOperation] = {}
    #     immutable_ops = [ImmutableOperation.from_op(op, context) for op in ops]
    #     return ImmutableBlock.from_immutable_ops(immutable_ops)

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
    parent_block: Optional[ImmutableBlock] = None

    @property
    def region(self):
        return self.regions[0]

    @property
    def result_types(self) -> List[Attribute]:
        return [result.typ for result in self.results]

    def __post_init__(self):
        for result in self.results:
            result.op = self
        for region in self.regions:
            region.parent_op = self
        self.operands.freeze()
        self.results.freeze()
        self.successors.freeze()
        self.regions.freeze()

    @classmethod
    def _create_new(
        cls,
        op_type: OperationType,
        immutable_operands: Optional[List[ImmutableSSAValue]] = None,
        result_types: Optional[List[Attribute]] = None,
        attributes: Optional[Dict[str, Attribute]] = None,
        successors: Optional[List[ImmutableBlock]] = None,
        regions: Optional[List[ImmutableRegion]] = None,
        value_map: Optional[Dict[SSAValue, SSAValue]] = None,
        block_map: Optional[Dict[Block, Block]] = None
    ) -> Tuple[List[ImmutableOperation], List[Operation]]:
        """Creates new mutable operations and returns an immutable view on them."""
        print("create new Op")

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
        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        dependant_imm_operations: List[ImmutableOperation] = []
        dependant_operations: List[Operation] = []
        operands = []

        for idx, imm_operand in enumerate(immutable_operands):
            # if imm_operand in value_map:
            #     immutable_operands[idx] = value_map[imm_operand]
            #     imm_operand = immutable_operands[idx]
            if isinstance(
                imm_operand, ImmutableOpResult) and (op := imm_operand.get_op(
                )) is not None and op.parent_block is not None:
                # parent block set means we have to clone the op
                clonedOps = ImmutableOperation._create_new(
                    op._op.__class__,
                    immutable_operands=list(op.operands),
                    result_types=[result.typ for result in op._op.results],
                    attributes=op._op.attributes.copy(),
                    successors=list(op.successors),
                    regions=list(op.regions),
                    value_map=value_map,
                    block_map=block_map)

                dependant_imm_operations.extend(clonedOps[0])
                dependant_operations.extend(clonedOps[1])
                operands.append(clonedOps[0][-1].results[
                    imm_operand.result_index].get_mutable())
            else:
                # if imm_operand in value_map:
                #     operands.append(value_map[imm_operand.get_mutable()])
                operands.append(imm_operand.get_mutable())

        # TODO: get Regions from the ImmutableRegions
        mutable_regions = []
        for region in regions:
            if region.parent_op is not None:
                # region has to be recreated
                mutable_regions.append(
                    ImmutableRegion._create_new(region, value_map,
                                                block_map)[1])
            else:
                mutable_regions.append(region._region)

        # This will not work properly for blocks where the arguments are used by ops inside
        # mutable_blocks.append(Block.from_arg_types([arg.typ for arg in block.args]))

        # successors is ImmutableBlock, not Block here!

        # the value map has to be used to update e.g. blockArguments here for Operation
        newOp: Operation = op_type.create(
            operands=list(operands),
            result_types=result_types,
            attributes=attributes,
            successors=[successor._block for successor in successors],
            regions=mutable_regions)

        return (dependant_imm_operations +
                [ImmutableOperation.from_op(newOp, value_map, block_map)
                 ]), dependant_operations + [newOp]

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
        return cls._create_new(op_type, immutable_operands, result_types,
                               attributes, successors, regions)[0]

    @staticmethod
    def from_op(
        op: Operation,
        value_map: Optional[Dict[SSAValue, ImmutableSSAValue]] = None,
        block_map: Optional[Dict[Block, ImmutableBlock]] = None
    ) -> ImmutableOperation:
        """creates an immutable view on an existing mutable op and all nested regions"""
        assert isinstance(op, Operation)
        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        operands: List[ImmutableSSAValue] = []
        for operand in op.operands:
            match operand:
                case OpResult():
                    operands.append(
                        ImmutableOpResult(
                            operand.typ,
                            value_map[operand].op  # type: ignore
                            if operand in value_map else
                            ImmutableOperation.from_op(operand.op),
                            operand.result_index))
                case BlockArgument():
                    if operand not in value_map:
                        raise Exception("Block argument expected in mapping")
                    operands.append(value_map[operand])
                case _:
                    raise Exception(
                        "Operand is expeected to be either OpResult or BlockArgument"
                    )
        results: List[ImmutableOpResult] = []
        for idx, result in enumerate(op.results):
            results.append(immutable_result := ImmutableOpResult(
                result.typ, None, result.result_index))  # type: ignore
            value_map[result] = immutable_result

        successors: List[ImmutableBlock] = []
        for successor in op.successors:
            if successor in block_map:
                successors.append(block_map[successor])
            else:
                newImmutableSuccessor = ImmutableBlock.from_block(successor)
                block_map[successor] = newImmutableSuccessor
                successors.append(newImmutableSuccessor)

        regions: List[ImmutableRegion] = []
        for region in op.regions:
            regions.append(ImmutableRegion.from_block_list(region.blocks))

        immutableOp = ImmutableOperation("immutable." + op.name, op,
                                         FrozenList(operands),
                                         FrozenList(results),
                                         FrozenList(successors),
                                         FrozenList(regions))

        return immutableOp

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