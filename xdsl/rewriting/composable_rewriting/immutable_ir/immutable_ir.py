from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from immutabledict import immutabledict

from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.utils.exceptions import InvalidIRException
from xdsl.utils.immutable_list import IList


@dataclass(frozen=True)
class ISSAValue(ABC):
    """
    Represents an immutable SSA variable. An immutable SSA variable is either an operation result
    or a basic block argument.
    """

    type: Attribute
    users: IList[IOperation]

    def _add_user(self, op: IOperation):
        self.users._unfreeze()  # pyright: ignore[reportPrivateUsage]
        self.users.append(op)
        self.users.freeze()

    def _remove_user(self, op: IOperation):
        if op not in self.users:
            raise Exception(
                f"Trying to remove a user ({op.name}) that is not an actual user of this value!"
            )

        self.users._unfreeze()  # pyright: ignore[reportPrivateUsage]
        self.users.remove(op)
        self.users.freeze()


@dataclass(frozen=True)
class IOpResult(ISSAValue):
    """Represents an immutable SSA variable defined by an operation result."""

    op: IOperation
    index: int

    def __hash__(self) -> int:
        return hash(id(self.op)) + hash(self.index)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IOpResult):
            return self.op == other.op and self.index == other.index
        return False


@dataclass(frozen=True)
class IBlockArg(ISSAValue):
    """Represents an immutable SSA variable defined by a basic block."""

    block: IBlock
    index: int

    def __hash__(self) -> int:
        return hash(id(self.block)) + hash(self.index)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return "BlockArg(type:" + self.type.name + ("attached") + ")"


@dataclass(frozen=True)
class IRegion:
    """An immutable region contains a CFG of immutable blocks. IRegions are contained in operations."""

    blocks: IList[IBlock]
    """Immutable blocks contained in the IRegion. The first block is the entry block."""

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, value: object) -> bool:
        return self is value

    @property
    def block(self) -> IBlock:
        """
        Returns the block of a single-block region.
        Returns an exception if the region is not single-block.
        """
        if len(self.blocks) != 1:
            raise ValueError(
                "'block' property of IRegion class is only available "
                "for single-block regions."
            )
        return self.blocks[0]

    @property
    def ops(self) -> IList[IOperation]:
        """
        Get the operations of a single-block region.
        Returns an exception if the region is not single-block.
        """
        if len(self.blocks) != 1:
            raise ValueError(
                "'ops' property of IRegion class is only available "
                "for single-block regions."
            )
        return self.block.ops

    def __init__(self, blocks: Sequence[IBlock]):
        """Creates a new immutable region from a sequence of immutable blocks."""

        object.__setattr__(self, "blocks", IList(blocks))
        self.blocks.freeze()

    @classmethod
    def from_mutable(
        cls,
        blocks: Iterable[Block],
        value_map: dict[SSAValue, ISSAValue] | None = None,
        block_map: dict[Block, IBlock] | None = None,
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
            block_map[block] = IBlock([], [])

        immutable_blocks = [
            IBlock.from_mutable(block, value_map, block_map) for block in blocks
        ]

        region = IRegion(immutable_blocks)

        # clean up successor references to blocks for ops inside this region
        for block, imm_block in zip(blocks, region.blocks):
            if block.parent is None:
                raise InvalidIRException(
                    "Cannot create an IRegion from a mutable Block "
                    "that is not attached to a Region."
                )
            dummy_block = block_map[block]
            for block in region.blocks:
                for op in block.ops:
                    if dummy_block in op.successors:
                        dummy_index = op.successors.index(dummy_block)
                        # replace dummy successor with actual successor
                        object.__setattr__(
                            op,
                            "successors",
                            IList(
                                op.successors[:dummy_index]
                                + [imm_block]
                                + op.successors[dummy_index + 1 :]
                            ),
                        )

        return region

    def to_mutable(
        self,
        value_mapping: dict[ISSAValue, SSAValue] | None = None,
        block_mapping: dict[IBlock, Block] | None = None,
    ) -> Region:
        """
        Returns a mutable region that is a copy of this immutable region.
        The value_mapping and block_mapping are used to map already known correspondings
        of immutable values to mutable values and immutable blocks to mutable blocks.
        """
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}
        mutable_blocks: list[Block] = []
        # All mutable blocks have to be initialized first so that ops can
        # refer to them in their successor lists.
        for block in self.blocks:
            mutable_blocks.append(mutable_block := Block(arg_types=block.arg_types))
            block_mapping[block] = mutable_block
        for block in self.blocks:
            # This will use the already created Block and populate it
            block.to_mutable(value_mapping=value_mapping, block_mapping=block_mapping)
        return Region(mutable_blocks)


@dataclass(frozen=True)
class IBlock:
    """An immutable block contains a list of immutable operations. IBlocks are contained in IRegions."""

    args: IList[IBlockArg]
    ops: IList[IOperation]

    @property
    def arg_types(self) -> list[Attribute]:
        frozen_arg_types = [arg.type for arg in self.args]
        return frozen_arg_types

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, value: object) -> bool:
        return self is value

    def __repr__(self) -> str:
        return (
            "block of" + str(len(self.ops)) + " operations with args: " + str(self.args)
        )

    def __post_init__(self):
        for arg in self.args:
            object.__setattr__(arg, "block", self)

    def __init__(
        self, args: Sequence[Attribute] | Sequence[IBlockArg], ops: Sequence[IOperation]
    ):
        """Creates a new immutable block."""

        # Type Guards:
        def is_iblock_arg_seq(list: Sequence[Any]) -> TypeGuard[Sequence[IBlockArg]]:
            return all([isinstance(elem, IBlockArg) for elem in list])

        def is_type_seq(list: Sequence[Any]) -> TypeGuard[Sequence[Attribute]]:
            return all([isinstance(elem, Attribute) for elem in list])

        block_args: Sequence[IBlockArg]
        if is_type_seq(args):
            block_args = [
                IBlockArg(type, IList(()), self, idx) for idx, type in enumerate(args)
            ]
        elif is_iblock_arg_seq(args):
            block_args = args
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
        value_map: dict[SSAValue, ISSAValue] | None = None,
        block_map: dict[Block, IBlock] | None = None,
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

        args: list[IBlockArg] = []
        for arg in block.args:
            # The IBlock that will house this IBlockArg is not constructed yet.
            # After construction the block field will be set by the IBlock.
            immutable_arg = IBlockArg(
                arg.type,
                IList(),
                cast(IBlock, None),
                arg.index,
            )
            args.append(immutable_arg)
            value_map[arg] = immutable_arg

        immutable_ops = [
            IOperation.from_mutable(
                op, value_map=value_map, block_map=block_map, existing_operands=None
            )
            for op in block.ops
        ]

        return IBlock(args, immutable_ops)

    def to_mutable(
        self,
        value_mapping: dict[ISSAValue, SSAValue] | None = None,
        block_mapping: dict[IBlock, Block] | None = None,
    ) -> Block:
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
            mutable_block = Block(arg_types=self.arg_types)
        for idx, arg in enumerate(self.args):
            value_mapping[arg] = mutable_block.args[idx]
        block_mapping[self] = mutable_block

        for immutable_op in self.ops:
            mutable_block.add_op(
                immutable_op.to_mutable(
                    value_mapping=value_mapping, block_mapping=block_mapping
                )
            )
        return mutable_block


def get_immutable_copy(op: Operation) -> IOperation:
    return IOperation.from_mutable(op)


@dataclass(frozen=True)
class IOperation:
    """Represents an immutable operation."""

    __match_args__ = ("op_type", "operands", "results", "successors", "regions")
    name: str
    op_type: type[Operation]
    properties: immutabledict[str, Attribute]
    attributes: immutabledict[str, Attribute]
    operands: IList[ISSAValue]
    results: IList[IOpResult]
    successors: IList[IBlock]
    regions: IList[IRegion]

    def __init__(
        self,
        name: str,
        op_type: type[Operation],
        attributes: immutabledict[str, Attribute],
        properties: immutabledict[str, Attribute],
        operands: Sequence[ISSAValue],
        result_types: Sequence[Attribute],
        successors: Sequence[IBlock],
        regions: Sequence[IRegion],
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "op_type", op_type)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "properties", properties)
        object.__setattr__(self, "operands", IList(operands))
        for operand in operands:
            operand._add_user(self)  # pyright: ignore[reportPrivateUsage]
        object.__setattr__(
            self,
            "results",
            IList(
                [
                    IOpResult(type, IList(()), self, idx)
                    for idx, type in enumerate(result_types)
                ]
            ),
        )
        object.__setattr__(self, "successors", IList(successors))
        object.__setattr__(self, "regions", IList(regions))

        self.operands.freeze()
        self.results.freeze()
        self.successors.freeze()
        self.regions.freeze()

    @classmethod
    def get(
        cls,
        name: str,
        op_type: type[Operation],
        operands: Sequence[ISSAValue],
        result_types: Sequence[Attribute],
        attributes: immutabledict[str, Attribute],
        properties: immutabledict[str, Attribute],
        successors: Sequence[IBlock],
        regions: Sequence[IRegion],
    ) -> IOperation:
        return cls(
            name,
            op_type,
            properties,
            attributes,
            operands,
            result_types,
            successors,
            regions,
        )

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, value: object, /) -> bool:
        return self is value

    @property
    def result(self) -> IOpResult:
        """
        Get the result of a of an IOperation with a single result.
        Returns an exception if the operation does not have exactly one result.
        """
        if len(self.results) != 1:
            raise ValueError(
                "'result' property of IOperation class is only available "
                "for IOperations with exactly one result."
            )
        return self.results[0]

    @property
    def region(self) -> IRegion:
        """
        Get the region of a of an IOperation with a single region.
        Returns an exception if the operation does not have exactly one region.
        """
        if len(self.regions) != 1:
            raise ValueError(
                "'region' property of IOperation class is only available "
                "for IOperations with exactly one region."
            )
        return self.regions[0]

    @property
    def result_types(self) -> list[Attribute]:
        return [result.type for result in self.results]

    def to_mutable(
        self,
        value_mapping: dict[ISSAValue, SSAValue] | None = None,
        block_mapping: dict[IBlock, Block] | None = None,
    ) -> Operation:
        """
        Returns a mutable operation that is a copy of this immutable operation.
        The value_mapping and block_mapping are used to map already known correspondings
        of immutable values to mutable values and immutable blocks to mutable blocks.
        """
        if value_mapping is None:
            value_mapping = {}
        if block_mapping is None:
            block_mapping = {}

        mutable_operands: list[SSAValue] = []
        for operand in self.operands:
            if operand in value_mapping:
                mutable_operands.append(value_mapping[operand])
            else:
                print(f"ERROR: op {self.name} uses SSAValue before definition")
                # Continuing to enable printing the IR including missing
                # operands for investigation
                mutable_operands.append(
                    OpResult(operand.type, cast(Operation, None), 0)
                )

        mutable_successors: list[Block] = []
        for successor in self.successors:
            if successor in block_mapping:
                mutable_successors.append(block_mapping[successor])
            else:
                raise InvalidIRException(
                    "Invalid IR: Block is not defined in the current region"
                )

        mutable_regions: list[Region] = []
        for region in self.regions:
            mutable_regions.append(
                region.to_mutable(
                    value_mapping=value_mapping, block_mapping=block_mapping
                )
            )

        new_op: Operation = self.op_type.create(
            operands=mutable_operands,
            result_types=[result.type for result in self.results],
            properties=dict(self.properties),
            attributes=dict(self.attributes),
            successors=mutable_successors,
            regions=mutable_regions,
        )

        # Add the results of this operation to the value mapping
        # so other operations can use them as operands.
        for idx, result in enumerate(self.results):
            m_result = new_op.results[idx]
            value_mapping[result] = m_result

        return new_op

    @classmethod
    def from_mutable(
        cls,
        op: Operation,
        value_map: dict[SSAValue, ISSAValue] | None = None,
        block_map: dict[Block, IBlock] | None = None,
        existing_operands: Sequence[ISSAValue] | None = None,
    ) -> IOperation:
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

        operands: list[ISSAValue] = []
        if existing_operands is None:
            for operand in op.operands:
                if isinstance(operand, OpResult):
                    if operand in value_map:
                        operands.append(value_map[operand])
                    else:
                        raise Exception("Operand used before definition")
                elif isinstance(operand, BlockArgument):
                    if operand not in value_map:
                        raise Exception(
                            "Block argument expected in mapping for op: " + op.name
                        )
                    operands.append(value_map[operand])
                else:
                    raise Exception(
                        "Operand is expected to be either OpResult or BlockArgument"
                    )
        else:
            operands.extend(existing_operands)

        properties: immutabledict[str, Attribute] = immutabledict(op.properties)
        attributes: immutabledict[str, Attribute] = immutabledict(op.attributes)

        successors: list[IBlock] = []
        for successor in op.successors:
            if successor in block_map:
                successors.append(block_map[successor])
            else:
                raise Exception(
                    "Successor not defined in current region, `from_mutable`\
                          probably has to be called on the parent operation."
                )

        regions: list[IRegion] = []
        for region in op.regions:
            regions.append(IRegion.from_mutable(region.blocks, value_map, block_map))

        immutable_op = IOperation.get(
            op.name,
            op_type,
            operands,
            op.result_types,
            properties,
            attributes,
            successors,
            regions,
        )

        for idx, result in enumerate(op.results):
            value_map[result] = immutable_op.results[idx]

        return immutable_op

    def get_attribute(self, name: str) -> Attribute:
        return self.attributes[name]

    def get_attributes_copy(self) -> dict[str, Attribute]:
        return dict(self.attributes)
