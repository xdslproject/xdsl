from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, TYPE_CHECKING, TypeVar, Set, Union
import typing
from frozenlist import FrozenList

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.parser import Parser
    from xdsl.printer import Printer

OperationType = TypeVar('OperationType', bound='Operation', covariant=True)


@dataclass
class MLContext:
    """Contains structures for operations/attributes registration."""
    _registeredOps: Dict[str,
                         typing.Type[Operation]] = field(default_factory=dict)
    _registeredAttrs: Dict[str, typing.Type[Attribute]] = field(
        default_factory=dict)

    def register_op(self, op: typing.Type[Operation]) -> None:
        """Register an operation definition. Operation names should be unique."""
        if op.name in self._registeredOps:
            raise Exception(f"Operation {op.name} has already been registered")
        self._registeredOps[op.name] = op

    def register_attr(self, attr: typing.Type[Attribute]) -> None:
        """Register an attribute definition. Attribute names should be unique."""
        if attr.name in self._registeredAttrs:
            raise Exception(
                f"Attribute {attr.name} has already been registered")
        self._registeredAttrs[attr.name] = attr

    def get_op(self, name: str) -> typing.Type[Operation]:
        """Get an operation class from its name."""
        if name not in self._registeredOps:
            raise Exception(f"Operation {name} is not registered")
        return self._registeredOps[name]

    def get_attr(self, name: str) -> typing.Type[Attribute]:
        """Get an attribute class from its name."""
        if name not in self._registeredAttrs:
            raise Exception(f"Attribute {name} is not registered")
        return self._registeredAttrs[name]


@dataclass(frozen=True)
class Use:
    """The use of a SSA value."""

    operation: Operation
    """The operation using the value."""

    index: int
    """The index of the operand using the value in the operation."""


@dataclass
class SSAValue(ABC):
    """A reference to an SSA variable.
    An SSA variable is either an operation result, or a basic block argument."""

    typ: Attribute
    """Each SSA variable is associated to a type."""

    uses: Set[Use] = field(init=False, default_factory=set)
    """All uses of the value."""

    @staticmethod
    def get(arg: SSAValue | Operation) -> SSAValue:
        """Get a new SSAValue from either a SSAValue, or an operation with a single result."""
        if isinstance(arg, SSAValue):
            return arg
        if isinstance(arg, Operation):
            if len(arg.results) == 1:
                return arg.results[0]
            raise ValueError(
                "SSAValue.build: expected operation with a single result.")
        raise TypeError(
            f"Expected SSAValue or Operation for SSAValue.get, but got {arg}")

    def add_use(self, use: Use):
        """Add a new use of the value."""
        self.uses.add(use)

    def remove_use(self, use: Use):
        """Remove a use of the value."""
        assert use in self.uses, "use to be removed was not in use list"
        self.uses.remove(use)

    def replace_by(self, value: SSAValue) -> None:
        """Replace the value by another value in all its uses."""
        for use in self.uses.copy():
            use.operation.replace_operand(use.index, value)
        assert len(self.uses) == 0, "unexpected error in xdsl"

    def erase(self, safe_erase: bool = True) -> None:
        """
        Erase the value.
        If safe_erase is True, then check that no operations use the value anymore.
        If safe_erase is False, then replace its uses by an ErasedSSAValue.
        """
        if safe_erase and len(self.uses) != 0:
            raise Exception(
                "Attempting to delete SSA value that still has uses.")
        self.replace_by(ErasedSSAValue(self.typ))


@dataclass
class OpResult(SSAValue):
    """A reference to an SSA variable defined by an operation result."""

    op: Operation
    """The operation defining the variable."""

    result_index: int
    """The index of the result in the defining operation."""

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class BlockArgument(SSAValue):
    """A reference to an SSA variable defined by a basic block argument."""

    block: Block
    """The block defining the variable."""

    index: int
    """The index of the variable in the block arguments."""

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class ErasedSSAValue(SSAValue):
    """
    An erased SSA variable.
    This is used during transformations when a SSA variable is destroyed but still used.
    """

    def __hash__(self):
        return hash(id(self))


AttrClass = TypeVar('AttrClass', bound='Attribute')


@dataclass(frozen=True)
class Attribute(ABC):
    """
    A compile-time value.
    Attributes are used to represent SSA variable types, and can be attached
    on operations to give extra information.
    """

    name: str = field(default="", init=False)
    """The attribute name should be a static field in the attribute classes."""

    @classmethod
    def build(cls: typing.Type[AttrClass], *args) -> AttrClass:
        """Create a new attribute using one of the builder defined in IRDL."""
        assert False


@dataclass(frozen=True)
class Data(Attribute):
    """An attribute represented by a Python structure."""

    @staticmethod
    @abstractmethod
    def parse(parser: Parser) -> Data:
        """Parse the attribute value."""
        ...

    @abstractmethod
    def print(self, printer: Printer) -> None:
        """Print the attribute value."""
        ...


@dataclass(frozen=True)
class ParametrizedAttribute(Attribute):
    """An attribute parametrized by other attributes."""

    name: str = field(default="", init=False)
    parameters: List[Attribute] = field(default_factory=list)

    def __post_init__(self):
        self.verify()

    def verify(self) -> None:
        ...


@dataclass
class Operation:
    """A generic operation. Operation definitions inherit this class."""

    name: str = field(default="", init=False)
    """The operation name. Should be a static member of the class"""

    _operands: FrozenList[SSAValue] = field(default_factory=FrozenList)
    """The operation operands."""

    results: List[OpResult] = field(default_factory=list)
    """The results created by the operation."""

    successors: List[Block] = field(default_factory=list)
    """
    The basic blocks that the operation may give control to.
    This list should be empty for non-terminator operations.
    """

    attributes: Dict[str, Attribute] = field(default_factory=dict)
    """The attributes attached to the operation."""

    regions: List[Region] = field(default_factory=list)
    """Regions arguments of the operation."""

    parent: Optional[Block] = field(default=None)
    """The block containing this operation."""

    @property
    def operands(self) -> FrozenList[SSAValue]:
        return self._operands

    @operands.setter
    def operands(self, new: Union[List[SSAValue], FrozenList[SSAValue]]):
        if isinstance(new, list):
            new = FrozenList(new)
        for idx, operand in enumerate(self._operands):
            operand.remove_use(Use(self, idx))
        for idx, operand in enumerate(new):
            operand.add_use(Use(self, idx))
        self._operands = new
        self._operands.freeze()

    def __post_init__(self):
        assert (self.name != "")
        assert (isinstance(self.name, str))

    @staticmethod
    def with_result_types(op: Any,
                          operands: Optional[List[SSAValue]] = None,
                          result_types: Optional[List[Attribute]] = None,
                          attributes: Optional[Dict[str, Attribute]] = None,
                          successors: Optional[List[Block]] = None,
                          regions: Optional[List[Region]] = None) -> Operation:

        operation = op()
        if operands is not None:
            for operand in operands:
                assert isinstance(
                    operand, SSAValue), "Operands must be of type SSAValue"
            operation.operands = operands
        if result_types is not None:
            operation.results = [
                OpResult(typ, operation, idx)
                for (idx, typ) in enumerate(result_types)
            ]
        if attributes is not None:
            operation.attributes = attributes
        if successors is not None:
            operation.successors = successors
        if regions is not None:
            for region in regions:
                operation.add_region(region)
        return operation

    @classmethod
    def create(cls: typing.Type[OperationType],
               operands: Optional[List[SSAValue]] = None,
               result_types: Optional[List[Attribute]] = None,
               attributes: Optional[Dict[str, Attribute]] = None,
               successors: Optional[List[Block]] = None,
               regions: Optional[List[Region]] = None) -> OperationType:
        return Operation.with_result_types(cls, operands, result_types,
                                           attributes, successors, regions)

    @classmethod
    def build(cls: typing.Type[OperationType],
              operands: List[Any] = None,
              result_types: List[Any] = None,
              attributes: Dict[str, Any] = None,
              successors: List[Any] = None,
              regions: List[Any] = None) -> OperationType:
        """Create a new operation using builders."""
        ...

    def replace_operand(self, operand_idx: int, new_operand: SSAValue) -> None:
        """Replace an operand with another operand."""
        self.operands = list(self._operands[:operand_idx]) + [
            new_operand
        ] + list(self._operands[operand_idx + 1:])

    def add_region(self, region: Region) -> None:
        """Add an unattached region to the operation."""
        if region.parent is not None:
            raise Exception(
                "Cannot add region that is already attached on an operation.")
        self.regions.append(region)
        region.parent = self

    def drop_all_references(self) -> None:
        """
        Drop all references to other operations.
        This function is called prior to deleting an operation.
        """
        self.parent = None
        for idx, operand in enumerate(self.operands):
            operand.remove_use(Use(self, idx))
        for region in self.regions:
            region.drop_all_references()

    def walk(self, fun: Callable[[Operation], None]) -> None:
        """Call a function on all operations contained in the operation (including this one)"""
        fun(self)
        for region in self.regions:
            region.walk(fun)

    def verify(self, verify_nested_ops: bool = True) -> None:
        for operand in self.operands:
            if isinstance(operand, ErasedSSAValue):
                raise Exception("Erased SSA value is used by the operation")
        self.verify_()
        if verify_nested_ops:
            for region in self.regions:
                region.verify()

    def verify_(self) -> None:
        pass

    def clone_without_regions(self: OperationType) -> OperationType:
        """Clone an operation, with empty regions instead."""
        operands = self.operands
        result_types = [res.typ for res in self.results]
        attributes = self.attributes.copy()
        successors = self.successors.copy()
        regions = [Region() for _ in self.regions]
        return self.create(operands=operands,
                           result_types=result_types,
                           attributes=attributes,
                           successors=successors,
                           regions=regions)

    def erase(self, safe_erase=True, drop_references=True) -> None:
        """
        Erase the operation, and remove all its references to other operations.
        If safe_erase is specified, check that the operation results are not used.
        """
        assert self.parent is None, "Operation with parents should first be detached before erasure."
        if drop_references:
            self.drop_all_references()
        for result in self.results:
            result.erase(safe_erase=safe_erase)

    def detach(self):
        """Detach the operation from its parent block."""
        if self.parent is None:
            raise Exception("Cannot detach a toplevel operation.")
        self.parent.detach_op(self)

    def get_toplevel_object(self) -> Union[Operation, Block, Region]:
        """Get the operation, block, or region ancestor that has no parents."""
        if self.parent is None:
            return self
        return self.parent.get_toplevel_object()

    def is_ancestor(self, op: Union[Operation, Block, Region]) -> bool:
        """Returns true if the operation is an ancestor of the operation, block, or region."""
        if op is self:
            return True
        if op.parent is None:
            return False
        return self.is_ancestor(op.parent)

    def __eq__(self, other: Operation) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(eq=False)
class Block:
    """A sequence of operations"""

    _args: FrozenList[BlockArgument] = field(default_factory=list, init=False)
    """The basic block arguments."""

    ops: List[Operation] = field(default_factory=list, init=False)
    """Ordered operations contained in the block."""

    parent: Optional[Region] = field(default=None, init=False)
    """Parent region containing the block."""

    @property
    def args(self) -> FrozenList[BlockArgument]:
        """Returns the block arguments."""
        return self._args

    @staticmethod
    def from_arg_types(arg_types: List[Attribute]) -> Block:
        b = Block()
        b._args = FrozenList([
            BlockArgument(typ, b, index) for index, typ in enumerate(arg_types)
        ])
        b._args.freeze()
        return b

    @staticmethod
    def from_ops(ops: List[Operation], arg_types: List[Attribute] = None):
        b = Block()
        if arg_types is not None:
            b._args = [
                BlockArgument(typ, b, index)
                for index, typ in enumerate(arg_types)
            ]
        b.add_ops(ops)
        return b

    @staticmethod
    def from_callable(block_arg_types: List[Attribute],
                      f: Callable[[BlockArgument, ...], List[Operation]]):
        b = Block.from_arg_types(block_arg_types)
        b.add_ops(f(*b.args))
        return b

    def is_ancestor(self, op: Union[Operation, Block, Region]) -> bool:
        """Returns true if the block is an ancestor of the operation, block, or region."""
        if op is self:
            return True
        if op.parent is None:
            return False
        return self.is_ancestor(op.parent)

    def insert_arg(self, typ: Attribute, index: int) -> BlockArgument:
        """
        Insert a new argument with a given type to the arguments list at a specific index.
        Returns the new argument.
        """
        if index < 0 or index > len(self._args):
            raise Exception("Unexpected index")
        new_arg = BlockArgument(typ, self, index)
        for arg in self._args[index:]:
            arg.index += 1
        self._args = FrozenList(
            list(self._args[:index]) + [new_arg] + list(self._args[index:]))
        self._args.freeze()
        return new_arg

    def erase_arg(self, arg: BlockArgument, safe_erase: bool = True) -> None:
        """
        Erase a block argument.
        If safe_erase is True, check that the block argument is not used.
        If safe_erase is False, replace the block argument uses with an ErasedSSAVAlue.
        """
        if arg.block is not self:
            raise Exception(
                "Attempting to delete an argument of the wrong block")
        for block_arg in self._args[arg.index + 1:]:
            block_arg.index -= 1
        self._args = FrozenList(
            list(self._args[:arg.index]) + list(self._args[arg.index + 1:]))
        arg.erase(safe_erase=safe_erase)

    def _attach_op(self, operation: Operation) -> None:
        """Attach an operation to the block, and check that it has no parents."""
        if operation.parent is not None:
            raise ValueError(
                "Can't add to a block an operation already attached to a block."
            )
        if operation.is_ancestor(self):
            raise ValueError(
                "Can't add an operation to a block contained in the operation."
            )
        operation.parent = self

    def add_op(self, operation: Operation) -> None:
        """
        Add an operation at the end of the block.
        The operation should not be attached to another block already.
        """
        self._attach_op(operation)
        self.ops.append(operation)

    def add_ops(self, ops: List[Operation]) -> None:
        """
        Add operations at the end of the block.
        The operations should not be attached to another block.
        """
        for op in ops:
            self.add_op(op)

    def insert_op(self, ops: Union[Operation, List[Operation]],
                  index: int) -> None:
        """
        Insert one or multiple operations at a given index in the block.
        The operations should not be attached to another block.
        """
        if index < 0 or index > len(self.ops):
            raise ValueError(
                f"Can't insert operation in index {index} in a block with {len(self.ops)} operations."
            )
        if not isinstance(ops, list):
            ops = [ops]
        for op in ops:
            self._attach_op(op)
        self.ops = self.ops[:index] + ops + self.ops[index:]

    def get_operation_index(self, op: Operation) -> int:
        """Get the operation position in a block."""
        if op.parent is not self:
            raise Exception("Operation is not a children of the block.")
        for idx, block_op in enumerate(self.ops):
            if block_op is op:
                return idx
        assert False, "Unexpected xdsl error"

    def detach_op(self, op: Union[int, Operation]) -> Operation:
        """
        Detach an operation from the block.
        Returns the detached operation.
        """
        if isinstance(op, Operation):
            op_idx = self.get_operation_index(op)
        else:
            op_idx = op
            op = self.ops[op_idx]
        if op.parent is not self:
            raise Exception("Cannot detach operation from a different block.")
        op.parent = None
        self.ops = self.ops[:op_idx] + self.ops[op_idx + 1:]
        return op

    def erase_op(self, op: Union[int, Operation], safe_erase=True) -> None:
        """
        Erase an operation from the block.
        If safe_erase is True, check that the operation has no uses.
        """
        op = self.detach_op(op)
        op.erase(safe_erase=safe_erase)

    def walk(self, fun: Callable[[Operation], None]) -> None:
        """Call a function on all operations contained in the block."""
        for op in self.ops:
            op.walk(fun)

    def verify(self) -> None:
        for operation in self.ops:
            if operation.parent != self:
                raise Exception(
                    "Parent pointer of operation does not refer to containing region"
                )
            operation.verify()

    def drop_all_references(self) -> None:
        """
        Drop all references to other operations.
        This function is called prior to deleting a block.
        """
        self.parent = None
        for op in self.ops:
            op.drop_all_references()

    def erase(self, safe_erase=True) -> None:
        """
        Erase the block, and remove all its references to other operations.
        If safe_erase is specified, check that no operation results are used outside the block.
        """
        assert self.parent is not None, "Blocks with parents should first be detached before erasure."
        self.drop_all_references()
        for op in self.ops:
            op.erase(safe_erase=safe_erase, drop_references=False)

    def get_toplevel_object(self) -> Union[Operation, Block, Region]:
        """Get the operation, block, or region ancestor that has no parents."""
        if self.parent is None:
            return self
        return self.parent.get_toplevel_object()

    def __eq__(self, other: Block) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class Region:
    """A region contains a CFG of blocks. Regions are contained in operations."""

    blocks: List[Block] = field(default_factory=list, init=False)
    """Blocks contained in the region. The first block is the entry block."""

    parent: Optional[Operation] = field(default=None, init=False)
    """Operation containing the region."""

    @staticmethod
    def from_operation_list(ops: List[Operation]) -> Region:
        block = Block.from_ops(ops)
        region = Region()
        region.add_block(block)
        return region

    @staticmethod
    def from_block_list(blocks: List[Block]) -> Region:
        region = Region()
        for block in blocks:
            region.add_block(block)
        return region

    @staticmethod
    def get(arg: Region | List[Block] | List[Operation]) -> Region:
        if isinstance(arg, Region):
            return arg
        if isinstance(arg, list):
            if len(arg) == 0:
                return Region.from_operation_list([])
            if isinstance(arg[0], Block):
                return Region.from_block_list(arg)
            if isinstance(arg[0], Operation):
                return Region.from_operation_list(arg)
        raise TypeError(f"Can't build a region with argument {arg}")

    @property
    def ops(self) -> List[Operation]:
        """
        Get the operations of a single-block region.
        Returns an exception if the region is not single-block.
        """
        if len(self.blocks) != 1:
            raise ValueError(
                "'ops' property of Region class is only available for single-block regions."
            )
        return self.blocks[0].ops

    @property
    def op(self) -> Operation:
        """
        Get the operation of a single-operation single-block region.
        Returns an exception if the region is not single-operation single-block.
        """
        if len(self.blocks) != 1 or len(self.blocks[0].ops) != 1:
            raise ValueError("'op' property of Region class is only available "
                             "for single-operation single-block regions.")
        return self.blocks[0].ops[0]

    def _attach_block(self, block: Block) -> None:
        """Attach a block to the region, and check that it has no parents."""
        if block.parent is not None:
            raise ValueError(
                "Can't add to a region a block already attached to a region.")
        if block.is_ancestor(self):
            raise ValueError(
                "Can't add a block to a region contained in the block.")
        block.parent = self

    def add_block(self, block: Block) -> None:
        """Add a block to the region."""
        self._attach_block(block)
        self.blocks.append(block)

    def insert_block(self, blocks: Union[Block, List[Block]],
                     index: int) -> None:
        """
        Insert one or multiple blocks at a given index in the region.
        The blocks should not be attached to another region.
        """
        if index < 0 or index > len(self.blocks):
            raise ValueError(
                f"Can't insert block in index {index} in a block with {len(self.blocks)} blocks."
            )
        if not isinstance(blocks, list):
            blocks = [blocks]
        for block in blocks:
            self._attach_block(block)
        self.blocks = self.blocks[:index] + blocks + self.blocks[index:]

    def get_block_index(self, block: Block) -> int:
        """Get the block position in a region."""
        if block.parent is not self:
            raise Exception("Block is not a child of the region.")
        for idx, region_block in enumerate(self.blocks):
            if region_block is block:
                return idx
        assert False, "Unexpected xdsl error"

    def detach_block(self, block: Union[int, Block]) -> Block:
        """
        Detach a block from the region.
        Returns the detached block.
        """
        if isinstance(block, Block):
            block_idx = self.get_block_index(block)
        else:
            block_idx = block
            op = self.blocks[block_idx]
        if block.parent is not self:
            raise Exception("Cannot detach block from a different region.")
        block.parent = None
        self.blocks = self.blocks[:block_idx] + self.blocks[block_idx + 1:]
        return block

    def erase_block(self, block: Union[int, Block], safe_erase=True) -> None:
        """
        Erase a block from the region.
        If safe_erase is True, check that the block has no uses.
        """
        block = self.detach_block(block)
        block.erase(safe_erase=safe_erase)

    def walk(self, fun: Callable[[Operation], None]) -> None:
        """Call a function on all operations contained in the region."""
        for block in self.blocks:
            block.walk(fun)

    def verify(self) -> None:
        for block in self.blocks:
            block.verify()
            if block.parent != self:
                raise Exception(
                    "Parent pointer of block does not refer to containing region"
                )

    def drop_all_references(self) -> None:
        """
        Drop all references to other operations.
        This function is called prior to deleting a region.
        """
        self.parent = None
        for block in self.blocks:
            block.drop_all_references()

    def erase(self) -> None:
        """
        Erase the region, and remove all its references to other operations.
        """
        assert self.parent is not None, "Regions with parents should first be detached before erasure."
        self.drop_all_references()

    def move_blocks(self, region: Region) -> None:
        """Move the blocks of this region to another region. Leave no blocks in this region."""
        region.blocks = self.blocks
        self.blocks = []
        for block in region.blocks:
            block.parent = region

    def get_toplevel_object(self) -> Union[Operation, Block, Region]:
        """Get the operation, block, or region ancestor that has no parents."""
        if self.parent is None:
            return self
        return self.parent.get_toplevel_object()

    def is_ancestor(self, op: Union[Operation, Block, Region]) -> bool:
        """Returns true if the region is an ancestor of the operation, block, or region."""
        if op is self:
            return True
        if op.parent is None:
            return False
        return self.is_ancestor(op.parent)
