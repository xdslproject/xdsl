from __future__ import annotations
from optparse import Option
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *
from xdsl.rewriter import Rewriter
from xdsl.printer import Printer


@dataclass(frozen=True)
class ImmutableSSAValue:
    typ: Attribute

    def get_op(self) -> ImmutableOperation:
        if isinstance(self, ImmutableOpResult):
            return self.op
        return None  # type: ignore

    # def __hash__(self):
    #     # return hash(self.typ)
    #     return hash(id(self))
    
    # def __eq__(self, __o: ImmutableSSAValue) -> bool:
    #     return self is __o

@dataclass(frozen=True)
class ImmutableOpResult(ImmutableSSAValue):
    op: ImmutableOperation
    result_index: int

    def __hash__(self):
        return hash(id(self.op)) + hash(self.result_index)
        # return hash((self.op, self.result_index))
    def __eq__(self, __o: ImmutableOpResult) -> bool:
        return self.op == __o.op and self.result_index == __o.result_index

@dataclass(frozen=True)
class ImmutableBlockArgument(ImmutableSSAValue):
    block: ImmutableBlock
    index: int

    def __hash__(self):
        return hash(id(self))
        # return hash((self.block, self.index))
    def __eq__(self, __o: ImmutableBlockArgument) -> bool:
        # return self.block == __o.block and self.index == __o.index
        return self is __o

    def __str__(self) -> str:
        return "BlockArg(type:" + self.typ.name + ("attached" if self.block is not None else "unattached") + ")"

    def __repr__(self) -> str:
        return "BlockArg(type:" + self.typ.name + ("attached" if self.block is not None else "unattached") + ")"
@dataclass(frozen=True)
class ImmutableRegion:
    blocks: FrozenList[ImmutableBlock]
    # parent_op: Optional[ImmutableOperation] = None

    def __hash__(self):
        return hash(id(self))
        # return hash((self.blocks))
    def __eq__(self, __o: ImmutableRegion) -> bool:
        return self is __o

    def __post_init__(self):
        # for block in self.blocks:
        #     block.parent_region = self
        self.blocks.freeze()

    @property
    def block(self):
        return self.blocks[0]

    def get_mutable_copy(self, value_mapping: Optional[Dict[ImmutableSSAValue, SSAValue]] = None, block_mapping: Optional[Dict[ImmutableBlock, Block]] = None) -> Region:
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}
        mutable_blocks: List[Block] = []
        for block in self.blocks:
            mutable_blocks.append(block.get_mutable_copy(value_mapping=value_mapping, block_mapping=block_mapping))
        return Region.from_block_list(mutable_blocks)

    @classmethod
    def create_new(
        cls,
        blocks: List[ImmutableBlock]
    ) -> ImmutableRegion:
        """Creates a new mutable region and returns an immutable view on it."""
        new_region = ImmutableRegion(FrozenList(blocks))
        return new_region

    @staticmethod
    def from_mutable(blocks: List[Block]) -> ImmutableRegion:
        immutable_blocks = [
            ImmutableBlock.from_mutable(block) for block in blocks
        ]
        assert (blocks[0].parent is not None)
        return ImmutableRegion(FrozenList(immutable_blocks))

    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
        for block in self.blocks:
            block.walk(fun)


@dataclass(frozen=True)
class ImmutableBlock:
    args: FrozenList[ImmutableBlockArgument]
    ops: FrozenList[ImmutableOperation]

    @property
    def arg_types(self) -> FrozenList[Attribute]:
        frozen_arg_types = FrozenList([arg.typ for arg in self.args])
        frozen_arg_types.freeze()
        return frozen_arg_types

    def __hash__(self):
        return(id(self))
        # return hash((self.ops)) # previously we also hashed the arg_types

    def __eq__(self, __o: ImmutableBlock) -> bool:
        return self is __o
        # return self.arg_types == __o.arg_types and self.ops == __o.ops

    def __str__(self) -> str:
        return "block of" + str(len(self.ops)) + " with args: " + str(self.args)

    def __repr__(self) -> str:
        return "block of" + str(len(self.ops)) + " with args: " + str(self.args)

    def __post_init__(self):
        for arg in self.args:
            object.__setattr__(arg, "block", self)

        self.args.freeze()
        self.ops.freeze()

    def get_mutable_copy(self, value_mapping: Optional[Dict[ImmutableSSAValue, SSAValue]] = None, block_mapping: Optional[Dict[ImmutableBlock, Block]] = None) -> Block:
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}

        new_block = Block.from_arg_types([arg.typ for arg in self.args])
        for idx, arg in enumerate(self.args):
            value_mapping[arg] = new_block.args[idx]
        block_mapping[self] = new_block

        for immutable_op in self.ops:
            new_block.add_op(immutable_op.get_mutable_copy(value_mapping=value_mapping, block_mapping=block_mapping))
        return new_block

    @classmethod
    def create_new(
            cls,
            arg_types: List[Attribute],
            ops: List[ImmutableOperation], environment: Optional[Dict[ImmutableSSAValue, ImmutableSSAValue]] = None, old_block: Optional[ImmutableBlock] = None) -> ImmutableBlock:
        """Creates a new immutable block."""
        if environment is None:
            environment = {}
        args: List[ImmutableBlockArgument] = []
        # If we have a reference to a preivous block and an existing mapping to new BlockArgs use them:
        if old_block is not None:
            assert(len(old_block.args) == len(arg_types))
            for idx, old_arg in enumerate(old_block.args):
                if old_arg in environment:
                    args.append(environment[old_arg])  # type: ignore
                else:
                    args.append(ImmutableBlockArgument(arg_types[idx], None, idx))  # type: ignore
                    print("Warning: assuming blockArg not used in block")
        else:
            args = [ImmutableBlockArgument(type, None, idx) for idx, type in enumerate(arg_types)] # type: ignore
        new_block = ImmutableBlock(FrozenList(args), FrozenList(ops))  # type: ignore
        return new_block

    @staticmethod
    def from_mutable(block: Block) -> ImmutableBlock:
        value_map: dict[SSAValue, ImmutableSSAValue] = {}
        block_map: dict[Block, ImmutableBlock] = {}

        args: List[ImmutableBlockArgument] = []
        for arg in block.args:
            immutable_arg = ImmutableBlockArgument(arg.typ, None, arg.index)  # type: ignore
            args.append(immutable_arg)
            value_map[arg] = immutable_arg

        immutable_ops = [
            ImmutableOperation.from_mutable(op,
                                       value_map=value_map,
                                       block_map=block_map) for op in block.ops
        ]

        return ImmutableBlock(FrozenList(args),
                              FrozenList(immutable_ops))

    @staticmethod
    def _from_args(block: Block, block_args: List[ImmutableBlockArgument],
                   imm_ops: List[ImmutableOperation]) -> ImmutableBlock:
        return ImmutableBlock(FrozenList(block_args),
                              FrozenList(imm_ops))


    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
        for op in self.ops:
            op.walk(fun)


def get_immutable_copy(op: Operation) -> ImmutableOperation:
    return ImmutableOperation.from_mutable(op, {})


@dataclass(frozen=True)
class ImmutableOperation:
    name: str
    op_type: type[Operation]
    operands: FrozenList[ImmutableSSAValue]
    results: FrozenList[ImmutableOpResult]
    attributes: Dict[str, Attribute]
    successors: FrozenList[ImmutableBlock]
    regions: FrozenList[ImmutableRegion]
    parent_block: Optional[ImmutableBlock] = None

    def __hash__(self) -> int:
        # return hash((self.name, self.attributes.values))
        return hash(id(self))

    # def __eq__(self, __o: ImmutableOperation) -> bool:
    #     return self.name == __o.name and self.op_type == __o.op_type and self.attributes == __o.attributes 
    def __eq__(self, __o: ImmutableOperation) -> bool:
        return self is __o

    @property
    def region(self):
        return self.regions[0]

    @property
    def result_types(self) -> List[Attribute]:
        return [result.typ for result in self.results]

    def __post_init__(self):
        for result in self.results:
            object.__setattr__(result, "op", self)
        self.operands.freeze()
        self.results.freeze()
        self.successors.freeze()
        self.regions.freeze()

    def get_mutable_copy(self, value_mapping: Optional[Dict[ImmutableSSAValue, SSAValue]] = None, block_mapping: Optional[Dict[ImmutableBlock, Block]] = None) -> Operation:
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}

        mutable_operands: List[SSAValue] = []
        for operand in self.operands:
            if operand in value_mapping:
                mutable_operands.append(value_mapping[operand])
            else:
                mutable_operands.append(operand)
                raise Exception("SSAValue used before definition")                  
                
        mutable_successors: List[Block] = []
        for successor in self.successors:
            if successor in block_mapping:
                mutable_successors.append(block_mapping[successor])
            else:
                raise Exception("Block used before definition")

        mutable_regions: List[Region] = []
        for region in self.regions:
            mutable_regions.append(region.get_mutable_copy(value_mapping=value_mapping, block_mapping=block_mapping))

        new_op: Operation = self.op_type.create(
            operands=mutable_operands,
            result_types=[result.typ for result in self.results],
            attributes=self.attributes.copy(),
            successors=mutable_successors,
            regions=mutable_regions)

        for idx, result in enumerate(self.results):
            m_result = new_op.results[idx]
            value_mapping[result] = m_result

        return new_op


    @classmethod
    def create_new(
        cls,
        op_type: type[Operation],
        operands: Optional[List[ImmutableSSAValue]] = None,
        result_types: Optional[List[Attribute]] = None,
        attributes: Optional[Dict[str, Attribute]] = None,
        successors: Optional[List[ImmutableBlock]] = None,
        regions: Optional[List[ImmutableRegion]] = None,
        environment: Optional[Dict[ImmutableSSAValue, ImmutableSSAValue]] = None
    ) -> Tuple[List[ImmutableOperation], Dict[ImmutableSSAValue, ImmutableSSAValue]]:

        if operands is None:
            operands = []
        if result_types is None:
            result_types = []
        if attributes is None:
            attributes = {}
        if successors is None:
            successors = []
        if regions is None:
            regions = []
        if environment is None:
            environment = {}

        remapped_operands = []
        for operand in operands:
            if isinstance(operand, ImmutableBlockArgument):
                if operand not in environment:
                    new_block_arg = ImmutableBlockArgument(operand.typ, None, operand.index)  # type: ignore
                    environment[operand] = new_block_arg
                remapped_operands.append(environment[operand])
            else:
                remapped_operands.append(operand)

        new_op = ImmutableOperation(op_type.name, op_type, FrozenList(remapped_operands), FrozenList([ImmutableOpResult(type, None, idx) for idx, type in enumerate(result_types)]), attributes, FrozenList(successors), FrozenList(regions))  # type: ignore
        return ([new_op], environment)

        # return cls._create_new(op_type, immutable_operands, result_types,
                            #    attributes, successors, regions)[0]

    @staticmethod
    def from_mutable(
        op: Operation,
        value_map: Optional[Dict[SSAValue, ImmutableSSAValue]] = None,
        block_map: Optional[Dict[Block, ImmutableBlock]] = None,
        existing_operands: Optional[List[ImmutableSSAValue]] = None
    ) -> ImmutableOperation:
        """creates an immutable view on an existing mutable op and all nested regions"""
        assert isinstance(op, Operation)
        op_type = op.__class__

        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        operands: List[ImmutableSSAValue] = []
        if existing_operands is None:
            for operand in op.operands:
                match operand:
                    case OpResult():
                        operands.append(
                            ImmutableOpResult(
                                operand.typ,
                                value_map[operand].op  # type: ignore
                                if operand in value_map else
                                ImmutableOperation.from_mutable(operand.op),
                                operand.result_index))
                    case BlockArgument():
                        if operand not in value_map:
                            raise Exception(
                                "Block argument expected in mapping")
                        operands.append(value_map[operand])
                    case _:
                        raise Exception(
                            "Operand is expeected to be either OpResult or BlockArgument"
                        )
        else:
            operands.extend(existing_operands)

        results: List[ImmutableOpResult] = []
        for idx, result in enumerate(op.results):
            results.append(immutable_result := ImmutableOpResult(
                result.typ, None, result.result_index))  # type: ignore
            value_map[result] = immutable_result

        attributes: Dict[str, Attribute] = op.attributes.copy()

        successors: List[ImmutableBlock] = []
        for successor in op.successors:
            if successor in block_map:
                successors.append(block_map[successor])
            else:
                newImmutableSuccessor = ImmutableBlock.from_mutable(successor)
                block_map[successor] = newImmutableSuccessor
                successors.append(newImmutableSuccessor)

        regions: List[ImmutableRegion] = []
        for region in op.regions:
            regions.append(ImmutableRegion.from_mutable(region.blocks))

        immutableOp = ImmutableOperation("immutable." + op.name,
                                         op_type,
                                         FrozenList(operands),
                                         FrozenList(results),
                                         attributes,
                                         FrozenList(successors),
                                         FrozenList(regions))

        return immutableOp

    def get_attribute(self, name: str) -> Attribute:
        return self.attributes[name]

    def get_attributes_copy(self) -> Dict[str, Attribute]:
        return self.attributes.copy()

    def walk(self, fun: Callable[[ImmutableOperation], None]) -> None:
        fun(self)
        for region in self.regions:
            region.walk(fun)   

def isa(op: Optional[ImmutableOperation], SomeOpClass: type[Operation]):
    if op is not None and op.op_type == SomeOpClass:
        return True
    else:
        return False