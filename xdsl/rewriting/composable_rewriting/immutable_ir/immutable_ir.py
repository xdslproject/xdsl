from __future__ import annotations
from abc import ABC
from typing import Iterable, Sequence, SupportsIndex, Type, TypeGuard, Any
from xdsl.ir import *
from xdsl.dialects.builtin import *
from xdsl.dialects.arith import *


_T = TypeVar('_T')
class IList(List[_T]):
    """
    A list that can be frozen. Once frozen, it can not be modified. 
    In comparison to FrozenList this supports pattern matching.
    """
    _frozen: bool = False

    def freeze(self):
        self._frozen = True

    def _unfreeze(self):
        self._frozen = False

    def append(self, __object: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().append(__object)

    def extend(self, __iterable: Iterable[_T]) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().extend(__iterable)

    def insert(self, __index: SupportsIndex, __object: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().insert(__index, __object)

    def remove(self, __value: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().remove(__value)

    def pop(self, __index: SupportsIndex = ...) -> _T:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().pop(__index)

    def clear(self) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().clear()


@dataclass(frozen=True)
class ISSAValue(ABC):
    """
    Represents an immutable SSA variable. An immutable SSA variable is either an operation result
    or a basic block argument.
    """
    typ: Attribute
    users: IList[IOp]

    def _add_user(self, op: IOp):
        self.users._unfreeze() # type: ignore
        self.users.append(op)
        self.users.freeze()

    def _remove_user(self, op: IOp):
        if op not in self.users:
            raise Exception(f"Trying to remove a user ({op.name}) that is not an actual user of this value!")

        self.users._unfreeze() # type: ignore
        self.users.remove(op)
        self.users.freeze()


@dataclass(frozen=True)
class IResult(ISSAValue):
    """Represents an immutable SSA variable defined by an operation result."""
    op: IOp
    result_index: int

    def __hash__(self) -> int:
        return hash(id(self.op)) + hash(self.result_index)

    def __eq__(self, __o: IResult) -> bool:
        if isinstance(__o, IResult):
            return self.op == __o.op and self.result_index == __o.result_index
        return False


@dataclass(frozen=True)
class IBlockArg(ISSAValue):
    """Represents an immutable SSA variable defined by a basic block."""
    block: IBlock
    index: int

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: IBlockArg) -> bool:
        return self is __o

    def __repr__(self) -> str:
        return "BlockArg(type:" + self.typ.name + (
            "attached" if self.block is not None else "unattached") + ")"


@dataclass(frozen=True)
class IRegion:
    """An immutable region contains a CFG of immutable blocks. IRegions are contained in operations."""

    blocks: IList[IBlock]
    """Immutable blocks contained in the IRegion. The first block is the entry block."""

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    @property
    def block(self) -> Optional[IBlock]:
        """Returns the entry block of this region."""

        if len(self.blocks) > 0:
            return self.blocks[0]
        return None

    @property
    def ops(self) -> IList[IOp]:
        """Returns a list of all operations in this region."""

        return IList([op for block in self.blocks for op in block.ops])
         
        
    def __init__(self, blocks: Sequence[IBlock]):
        """Creates a new immutable region from a sequence of immutable blocks."""

        object.__setattr__(self, "blocks", IList(blocks))
        self.blocks.freeze()

    @classmethod
    def from_mutable(
        cls,
        blocks: Sequence[Block],
        value_map: Optional[dict[SSAValue, ISSAValue]] = None,
        block_map: Optional[dict[Block, IBlock]] = None,
    ) -> IRegion:
        """
        Creates a new immutable region from a sequence of mutable blocks.
        The value_map and block_map are used to map already known correspondings
        of mutable values to immutable values and mutable blocks to immutable blocks.
        """
        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        # adding dummy block mappings so that ops have a successor to reference
        # when the actual block is created all successor references will be moved
        # to the correct block
        for block in blocks:
            block_map[block] = IBlock([],[])

        immutable_blocks = [
            IBlock.from_mutable(block, value_map, block_map)
            for block in blocks
        ]

        assert (blocks[0].parent is not None)
        region = IRegion(immutable_blocks)

        # clean up successor references to blocks for ops inside this region
        for block, imm_block in zip(blocks, region.blocks):
            dummy_block = block_map[block]
            for op in region.ops:
                try:
                    dummy_index = op.successors.index(dummy_block)
                except ValueError:
                    continue
                # replace dummy successor with actual successor
                object.__setattr__(op, "successors", IList(op.successors[:dummy_index]+[imm_block]+op.successors[dummy_index+1:]))

        return region

    def get_mutable_copy(
            self,
            value_mapping: Optional[dict[ISSAValue, SSAValue]] = None,
            block_mapping: Optional[dict[IBlock, Block]] = None) -> Region:
        """
        Returns a mutable region that is a copy of this immutable region.
        The value_mapping and block_mapping are used to map already known correspondings
        of immutable values to mutable values and immutable blocks to mutable blocks.
        """
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}
        mutable_blocks: List[Block] = []
        # All mutable blocks have to be initialized first so that ops can
        # refer to them in their successor lists.
        for block in self.blocks:
            mutable_blocks.append(mutable_block := Block.from_arg_types(block.arg_types))
            block_mapping[block] = mutable_block
        for block in self.blocks:
            # This will use the already created Block and populate it
            block.get_mutable_copy(value_mapping=value_mapping, block_mapping=block_mapping)
        return Region.from_block_list(mutable_blocks)


@dataclass(frozen=True)
class IBlock:
    """An immutable block contains a list of immutable operations. IBlocks are contained in IRegions."""
    args: IList[IBlockArg]
    ops: IList[IOp]

    @property
    def arg_types(self) -> List[Attribute]:
        frozen_arg_types = [arg.typ for arg in self.args]
        return frozen_arg_types

    def __hash__(self) -> int:
        return (id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    def __repr__(self) -> str:
        return "block of" + str(len(self.ops)) + " with args: " + str(
            self.args)

    def __post_init__(self):
        for arg in self.args:
            object.__setattr__(arg, "block", self)

    def __init__(self, args: Sequence[Attribute] | Sequence[IBlockArg],
                 ops: Sequence[IOp]):
        """Creates a new immutable block."""

        # Type Guards:
        def is_iblock_arg_seq(
                list: Sequence[Any]) -> TypeGuard[Sequence[IBlockArg]]:
            if len(list) == 0:
                return False
            return all([isinstance(elem, IBlockArg) for elem in list])

        def is_type_seq(list: Sequence[Any]) -> TypeGuard[Sequence[Attribute]]:
            return all([isinstance(elem, Attribute) for elem in list])

        if is_type_seq(args):
            block_args: Sequence[IBlockArg] = [
                IBlockArg(type, IList([]), self, idx) for idx, type in enumerate(args)
            ]
        elif is_iblock_arg_seq(args):
            block_args: Sequence[IBlockArg] = args
            for block_arg in block_args:
                object.__setattr__(block_arg, "block", self)
        else:
            raise Exception("args for IBlock ill structured")

        object.__setattr__(self, "args", IList(block_args))
        object.__setattr__(self, "ops", IList(ops))

        self.args.freeze()
        self.ops.freeze()

    @classmethod
    def from_mutable(
        cls,
        block: Block,
        value_map: Optional[dict[SSAValue, ISSAValue]] = None,
        block_map: Optional[dict[Block, IBlock]] = None,
    ) -> IBlock:
        """
        Creates an immutable block from a mutable block. 
        The value_map and block_map are used to map already known correspondings
        of mutable values to immutable values and mutable blocks to immutable blocks.
        """
        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        args: List[IBlockArg] = []
        for arg in block.args:
            # The IBlock that will house this IBlockArg is not constructed yet.
            # After construction the block field will be set by the IBlock.
            immutable_arg = IBlockArg(arg.typ, IList([]), None, arg.index)  # type: ignore
            args.append(immutable_arg)
            value_map[arg] = immutable_arg

        immutable_ops = [
            IOp.from_mutable(op, value_map=value_map, block_map=block_map, existing_operands=None)
            for op in block.ops
        ]

        return IBlock(args, immutable_ops)

    def get_mutable_copy(
            self,
            value_mapping: Optional[dict[ISSAValue, SSAValue]] = None,
            block_mapping: Optional[dict[IBlock, Block]] = None) -> Block:
        """
        Returns a mutable block that is a copy of this immutable block.
        The value_mapping and block_mapping are used to map already known correspondings
        of immutable values to mutable values and immutable blocks to mutable blocks.
        """
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}


        # Block might already have been created by the Region, look it up
        if self in block_mapping:
            mutable_block = block_mapping[self]
        else:
            mutable_block = Block.from_arg_types(self.arg_types)
        for idx, arg in enumerate(self.args):
            value_mapping[arg] = mutable_block.args[idx]
        block_mapping[self] = mutable_block

        for immutable_op in self.ops:
            mutable_block.add_op(
                immutable_op.get_mutable_copy(value_mapping=value_mapping,
                                              block_mapping=block_mapping))
        return mutable_block


def get_immutable_copy(op: Operation) -> IOp:
    return IOp.from_mutable(op, {})


@dataclass(frozen=True)
class OpData:
    """
    Represents split off fields of IOp to its own class because they are
    often preserved during rewriting. A new operation of the same type, e.g.
    with changed operands can still use the same OpData instance. This design
    increases sharing in the IR.
    """
    name: str
    op_type: type[Operation]
    attributes: dict[str, Attribute]


@dataclass(frozen=True)
class IOp:
    """Represents an immutable operation."""
    
    __match_args__ = ("op_type", "operands", "results", "successors",
                      "regions")
    _op_data: OpData
    operands: IList[ISSAValue]
    results: IList[IResult]
    successors: IList[IBlock]
    regions: IList[IRegion]

    def __init__(self, op_data: OpData, operands: Sequence[ISSAValue],
                 result_types: Sequence[Attribute],
                 successors: Sequence[IBlock],
                 regions: Sequence[IRegion]) -> None:
        object.__setattr__(self, "_op_data", op_data)
        object.__setattr__(self, "operands", IList(operands))
        for operand in operands:
            operand._add_user(self) # type: ignore
        object.__setattr__(
            self, "results",
            IList([
                IResult(type, IList([]), self, idx)
                for idx, type in enumerate(result_types)
            ]))
        object.__setattr__(self, "successors", IList(successors))
        object.__setattr__(self, "regions", IList(regions))

        self.operands.freeze()
        self.results.freeze()
        self.successors.freeze()
        self.regions.freeze()

    @classmethod
    def get(cls, name: str, op_type: type[Operation],
            operands: Sequence[ISSAValue], result_types: Sequence[Attribute],
            attributes: dict[str, Attribute], successors: Sequence[IBlock],
            regions: Sequence[IRegion]) -> IOp:
        return cls(OpData(name, op_type, attributes), operands, result_types,
                   successors, regions)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    @property
    def name(self) -> str:
        return self._op_data.name

    @property
    def op_type(self) -> Type[Operation]:
        return self._op_data.op_type

    @property
    def attributes(self) -> dict[str, Attribute]:
        return self._op_data.attributes

    @property
    def result(self) -> IResult | None:
        if len(self.results) > 0:
            return self.results[0]
        return None

    @property
    def region(self) -> IRegion | None:
        if len(self.regions) > 0:
            return self.regions[0]
        return None

    @property
    def result_types(self) -> List[Attribute]:
        return [result.typ for result in self.results]

    def get_mutable_copy(
            self,
            value_mapping: Optional[dict[ISSAValue, SSAValue]] = None,
            block_mapping: Optional[dict[IBlock, Block]] = None) -> Operation:
        """
        Returns a mutable operation that is a copy of this immutable operation.
        The value_mapping and block_mapping are used to map already known correspondings
        of immutable values to mutable values and immutable blocks to mutable blocks.
        """
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}

        mutable_operands: List[SSAValue] = []
        for operand in self.operands:
            if operand in value_mapping:
                mutable_operands.append(value_mapping[operand])
            else:
                print(f"ERROR: op {self.name} uses SSAValue before definition")
                # Continuing to enable printing the IR including missing
                # operands for investigation
                mutable_operands.append(
                    OpResult(
                        operand.typ,
                        None,  # type: ignore
                        0))

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
    def from_mutable(
            cls,
            op: Operation,
            value_map: Optional[dict[SSAValue, ISSAValue]] = None,
            block_map: Optional[dict[Block, IBlock]] = None,
            existing_operands: Optional[Sequence[ISSAValue]] = None) -> IOp:
        """
        Returns an immutable operation that is a copy of the given mutable operation.
        The value_map and block_map are used to map already known correspondings
        of mutable values to immutable values and mutable blocks to immutable blocks.
        """
        assert isinstance(op, Operation)
        op_type = op.__class__

        if value_map is None:
            value_map = {}
        if block_map is None:
            block_map = {}

        operands: List[ISSAValue] = []
        if existing_operands is None:
            for operand in op.operands:
                match operand:
                    case OpResult():
                        if operand in value_map:
                            operands.append(value_map[operand])
                        else:
                            raise Exception("Operand used before definition")
                    case BlockArgument():
                        if operand not in value_map:
                            raise Exception(
                                "Block argument expected in mapping for op: " +
                                op.name)
                        operands.append(value_map[operand])
                    case _:
                        raise Exception(
                            "Operand is expected to be either OpResult or BlockArgument"
                        )
        else:
            operands.extend(existing_operands)

        attributes: dict[str, Attribute] = op.attributes.copy()

        successors: List[IBlock] = []
        for successor in op.successors:
            if successor in block_map:
                successors.append(block_map[successor])
            else:
                # TODO: I think this is not right, build tests with successors
                newImmutableSuccessor = IBlock.from_mutable(successor)
                block_map[successor] = newImmutableSuccessor
                successors.append(newImmutableSuccessor)

        regions: List[IRegion] = []
        for region in op.regions:
            regions.append(
                IRegion.from_mutable(region.blocks, value_map, block_map))

        immutable_op = IOp.get(op.name, op_type, operands,
                               [result.typ for result in op.results],
                               attributes, successors, regions)

        for idx, result in enumerate(op.results):
            value_map[result] = immutable_op.results[idx]

        return immutable_op

    def get_attribute(self, name: str) -> Attribute:
        return self.attributes[name]

    def get_attributes_copy(self) -> dict[str, Attribute]:
        return self.attributes.copy()