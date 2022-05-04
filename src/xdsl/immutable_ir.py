from __future__ import annotations
from optparse import Option
from typing import Generic
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *
from xdsl.rewriter import Rewriter
from xdsl.printer import Printer

T = TypeVar('T')


# TODO: Does it make more sense to inherit from MutableSequence
# and implement the frozen aspect ourselves?
# potentially we have the items in memory twice here?
class IList(FrozenList[T]):
    __match_args__ = ('_items', )
    _items: List

    def __init__(self, items=None):
        if items is not None:
            items = list(items)
        else:
            items = []
        self._items = items
        super(IList, self).__init__(items)


@dataclass(frozen=True)
class IVal:
    typ: Attribute


@dataclass(frozen=True)
class IRes(IVal):
    op: IOp
    result_index: int

    def __hash__(self):
        return hash(id(self.op)) + hash(self.result_index)

    def __eq__(self, __o: IRes) -> bool:
        return self.op == __o.op and self.result_index == __o.result_index


@dataclass(frozen=True)
class IBlockArg(IVal):
    block: IBlock
    index: int

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, __o: IBlockArg) -> bool:
        return self is __o

    def __str__(self) -> str:
        return "BlockArg(type:" + self.typ.name + (
            "attached" if self.block is not None else "unattached") + ")"

    def __repr__(self) -> str:
        return "BlockArg(type:" + self.typ.name + (
            "attached" if self.block is not None else "unattached") + ")"


@dataclass(frozen=True)
class IRegion:
    blocks: IList[IBlock]

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, __o: IRegion) -> bool:
        return self is __o

    @property
    def block(self):
        return self.blocks[0]

    def __init__(self, blocks: List[IBlock]):
        """Creates a new mutable region and returns an immutable view on it."""
        object.__setattr__(self, "blocks", IList(blocks))
        self.blocks.freeze()

    @classmethod
    def from_mutable(cls, blocks: List[Block]) -> IRegion:
        immutable_blocks = [IBlock.from_mutable(block) for block in blocks]
        assert (blocks[0].parent is not None)
        return IRegion(immutable_blocks)

    def get_mutable_copy(
            self,
            value_mapping: Optional[Dict[IVal, SSAValue]] = None,
            block_mapping: Optional[Dict[IBlock, Block]] = None) -> Region:
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}
        mutable_blocks: List[Block] = []
        for block in self.blocks:
            mutable_blocks.append(
                block.get_mutable_copy(value_mapping=value_mapping,
                                       block_mapping=block_mapping))
        return Region.from_block_list(mutable_blocks)

    def walk(self, fun: Callable[[IOp], None]) -> None:
        for block in self.blocks:
            block.walk(fun)


@dataclass(frozen=True)
class IBlock:
    args: IList[IBlockArg]
    ops: IList[IOp]

    @property
    def arg_types(self) -> IList[Attribute]:
        frozen_arg_types = IList([arg.typ for arg in self.args])
        frozen_arg_types.freeze()
        return frozen_arg_types

    def __hash__(self):
        return (id(self))

    def __eq__(self, __o: IBlock) -> bool:
        return self is __o

    def __str__(self) -> str:
        return "block of" + str(len(self.ops)) + " with args: " + str(
            self.args)

    def __repr__(self) -> str:
        return "block of" + str(len(self.ops)) + " with args: " + str(
            self.args)

    def __post_init__(self):
        for arg in self.args:
            object.__setattr__(arg, "block", self)
        self.args.freeze()
        self.ops.freeze()

    def __init__(self,
                 args: Union[List[Attribute], List[IBlockArg]],
                 ops: List[IOp],
                 environment: Optional[Dict[IVal, IVal]] = None,
                 old_block: Optional[IBlock] = None):
        """Creates a new immutable block."""
        if environment is None:
            environment = {}

        # use typeguard
        if all([isinstance(arg, IBlockArg) for arg in args]):
            block_args: List[IBlockArg] = args
        else:
            block_args: List[IBlockArg] = []
            if old_block is not None:
                assert (len(old_block.args) == len(args))
                for idx, old_arg in enumerate(old_block.args):
                    if old_arg in environment:
                        assert isinstance(
                            old_block_arg := environment[old_arg], IBlockArg)
                        block_args.append(old_block_arg)
                    else:
                        block_args.append(IBlockArg(args[idx], self, idx))
                        print("Warning: assuming blockArg not used in block")
            else:
                block_args = [
                    IBlockArg(type, self, idx) for idx, type in enumerate(args)
                ]

        object.__setattr__(self, "args", IList(block_args))
        object.__setattr__(self, "ops", IList(ops))

        self.args.freeze()
        self.ops.freeze()

    @classmethod
    def from_mutable(cls, block: Block) -> IBlock:
        value_map: dict[SSAValue, IVal] = {}
        block_map: dict[Block, IBlock] = {}

        args: List[IBlockArg] = []
        for arg in block.args:
            immutable_arg = IBlockArg(arg.typ, None, arg.index)  # type: ignore
            args.append(immutable_arg)
            value_map[arg] = immutable_arg

        immutable_ops = [
            IOp.from_mutable(op, value_map=value_map, block_map=block_map)
            for op in block.ops
        ]

        return IBlock(args, immutable_ops)

    def get_mutable_copy(
            self,
            value_mapping: Optional[Dict[IVal, SSAValue]] = None,
            block_mapping: Optional[Dict[IBlock, Block]] = None) -> Block:
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}

        new_block = Block.from_arg_types([arg.typ for arg in self.args])
        for idx, arg in enumerate(self.args):
            value_mapping[arg] = new_block.args[idx]
        block_mapping[self] = new_block

        for immutable_op in self.ops:
            new_block.add_op(
                immutable_op.get_mutable_copy(value_mapping=value_mapping,
                                              block_mapping=block_mapping))
        return new_block

    def walk(self, fun: Callable[[IOp], None]) -> None:
        for op in self.ops:
            op.walk(fun)


def get_immutable_copy(op: Operation) -> IOp:
    return IOp.from_mutable(op, {})


@dataclass(frozen=True)
class OpData:
    name: str
    op_type: type[Operation]
    attributes: Dict[str, Attribute]


@dataclass(frozen=True)
class IOp:
    # __match_args__ = ("op_type", "operands", "results", "successors",
    #                   "regions")
    _op_data: OpData
    operands: IList[IVal]
    results: IList[IRes]
    successors: IList[IBlock]
    regions: IList[IRegion]
    parent_block: Optional[IList[IBlock]] = None

    def __init__(self, name: str, op_type: type[Operation],
                 operands: List[IVal], result_types: List[Attribute],
                 attributes: Dict[str, Attribute], successors: List[IBlock],
                 regions: List[IRegion]) -> None:
        object.__setattr__(self, "_op_data", OpData(name, op_type, attributes))
        object.__setattr__(self, "operands", IList(operands))
        object.__setattr__(
            self,
            "results",
            IList([
                IRes(type, self, idx)  # type: ignore
                for idx, type in enumerate(result_types)
            ]))
        object.__setattr__(self, "successors", IList(successors))
        object.__setattr__(self, "regions", IList(regions))

        self.operands.freeze()
        self.results.freeze()
        self.successors.freeze()
        self.regions.freeze()

        for result in self.results:
            object.__setattr__(result, "op", self)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: IOp) -> bool:
        return self is __o

    @property
    def name(self):
        return self._op_data.name

    @property
    def op_type(self):
        return self._op_data.op_type

    @property
    def attributes(self):
        return self._op_data.attributes

    @property
    def result(self):
        return self.results[0]

    @property
    def region(self):
        return self.regions[0]

    @property
    def result_types(self) -> List[Attribute]:
        return [result.typ for result in self.results]

    def get_mutable_copy(
            self,
            value_mapping: Optional[Dict[IVal, SSAValue]] = None,
            block_mapping: Optional[Dict[IBlock, Block]] = None) -> Operation:
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}

        mutable_operands: List[SSAValue] = []
        for operand in self.operands:
            if operand in value_mapping:
                mutable_operands.append(value_mapping[operand])
            else:
                raise Exception("SSAValue used before definition")

        mutable_successors: List[Block] = []
        for successor in self.successors:
            if successor in block_mapping:
                mutable_successors.append(block_mapping[successor])
            else:
                raise Exception("Block used before definition")

        mutable_regions: List[Region] = []
        for region in self.regions:
            mutable_regions.append(
                region.get_mutable_copy(value_mapping=value_mapping,
                                        block_mapping=block_mapping))

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
    def from_mutable(cls,
                     op: Operation,
                     value_map: Optional[Dict[SSAValue, IVal]] = None,
                     block_map: Optional[Dict[Block, IBlock]] = None,
                     existing_operands: Optional[List[IVal]] = None) -> IOp:
        """creates an immutable view on an existing mutable op and all nested regions"""
        assert isinstance(op, Operation)
        op_type = op.__class__

        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        operands: List[IVal] = []
        if existing_operands is None:
            for operand in op.operands:
                match operand:
                    case OpResult():
                        operands.append(
                            IRes(
                                operand.typ,
                                value_map[operand].op  # type: ignore
                                if operand in value_map else IOp.from_mutable(
                                    operand.op),
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

        attributes: Dict[str, Attribute] = op.attributes.copy()

        successors: List[IBlock] = []
        for successor in op.successors:
            if successor in block_map:
                successors.append(block_map[successor])
            else:
                newImmutableSuccessor = IBlock.from_mutable(successor)
                block_map[successor] = newImmutableSuccessor
                successors.append(newImmutableSuccessor)

        regions: List[IRegion] = []
        for region in op.regions:
            regions.append(IRegion.from_mutable(region.blocks))

        immutable_op = IOp("immutable." + op.name, op_type, operands,
                           [result.typ for result in op.results], attributes,
                           successors, regions)

        for idx, result in enumerate(op.results):
            value_map[result] = immutable_op.results[idx]

        return immutable_op

    def get_attribute(self, name: str) -> Attribute:
        return self.attributes[name]

    def get_attributes_copy(self) -> Dict[str, Attribute]:
        return self.attributes.copy()

    def walk(self, fun: Callable[[IOp], None]) -> None:
        fun(self)
        for region in self.regions:
            region.walk(fun)


class IBuilder:

    environment: Dict[IVal, IVal] = {}
    last_op_created: Optional[IOp] = None

    def op(self,
           op_type: type[Operation],
           operands: Optional[List[Union[IVal, IOp]]] = None,
           result_types: Optional[List[Attribute]] = None,
           attributes: Optional[Dict[str, Attribute]] = None,
           successors: Optional[List[IBlock]] = None,
           regions: Optional[List[IRegion]] = None) -> IOp:
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

        remapped_operands = []
        for operand in operands:
            if isinstance(operand, IOp):
                assert (len(operand.results) > 0)
                operand = operand.results[0]
            if isinstance(operand, IBlockArg):
                if operand not in self.environment:
                    new_block_arg = IBlockArg(
                        operand.typ,
                        None,  # type: ignore
                        operand.index)
                    self.environment[operand] = new_block_arg
                remapped_operands.append(self.environment[operand])
            else:
                remapped_operands.append(operand)

        new_op = IOp(op_type.name, op_type, remapped_operands, result_types,
                     attributes, successors, regions)
        self.last_op_created = new_op
        return new_op

    def from_op(self,
                old_op: IOp,
                operands: Optional[List[Union[IVal, IOp]]] = None,
                result_types: Optional[List[Attribute]] = None,
                attributes: Optional[Dict[str, Attribute]] = None,
                successors: Optional[List[IBlock]] = None,
                regions: Optional[List[IRegion]] = None):

        if operands is None:
            operands = list(old_op.operands)
        if result_types is None:
            result_types = list(old_op.result_types)
        if attributes is None:
            attributes = old_op.attributes
        if successors is None:
            successors = list(old_op.successors)
        if regions is None:
            regions = list(old_op.regions)

        return self.op(old_op.op_type, operands, result_types, attributes,
                       successors, regions)
