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
        if name not in self._registeredOps:
            raise Exception(f"Operation {name} is not registered")
        return self._registeredOps[name]

    def get_attr(self, name: str) -> typing.Type[Attribute]:
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


@dataclass
class OpResult(SSAValue):
    """A reference to an SSA variable defined by an operation result."""

    op: Operation
    """The operation defining the variable."""

    result_index: int
    """The index of the result in the defining operation."""

    def __hash__(self):
        return hash((id(self.op), self.result_index))


@dataclass
class BlockArgument(SSAValue):
    """A reference to an SSA variable defined by a basic block argument."""

    block: Block
    """The block defining the variable."""

    index: int
    """The index of the variable in the block arguments."""

    def __hash__(self):
        return hash((id(self.block), self.index))


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
              operands=[],
              result_types=[],
              attributes=[],
              successors=[],
              regions=[]) -> OperationType:
        """Create a new operation using builders."""
        ...

    def add_region(self, region: Region) -> None:
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

    def verify(self) -> None:
        self.verify_()
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
        assert self.parent is not None, "Operation with parents should first be detached before erasure."
        if drop_references:
            self.drop_all_references()
        if safe_erase:
            for result in self.results:
                assert len(result.uses) == 0

    def __eq__(self, other: Operation) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(eq=False)
class Block:
    """A sequence of operations"""

    args: List[BlockArgument] = field(default_factory=list, init=False)
    """The basic block arguments."""

    ops: List[Operation] = field(default_factory=list, init=False)
    """Ordered operations contained in the block."""

    parent: Optional[Region] = field(default=None, init=False)
    """Parent region containing the block."""

    @staticmethod
    def from_arg_types(arg_types: List[Attribute]) -> Block:
        b = Block()
        b.args = [
            BlockArgument(typ, b, index) for index, typ in enumerate(arg_types)
        ]
        return b

    @staticmethod
    def from_ops(ops: List[Operation], arg_types: List[Attribute] = None):
        b = Block()
        if arg_types is not None:
            b.args = [
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

    def add_op(self, operation: Operation) -> None:
        self.ops.append(operation)
        operation.parent = self

    def add_ops(self, ops: List[Operation]) -> None:
        for op in ops:
            self.add_op(op)

    def walk(self, fun: Callable[[Operation], None]) -> None:
        """Call a function on all operations contained in the block."""
        for op in self.ops:
            op.walk(fun)

    def verify(self) -> None:
        for operation in self.ops:
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

    def add_block(self, block: Block) -> None:
        self.blocks.append(block)
        block.parent = self

    def walk(self, fun: Callable[[Operation], None]) -> None:
        """Call a function on all operations contained in the region."""
        for block in self.blocks:
            block.walk(fun)

    def verify(self) -> None:
        for block in self.blocks:
            block.verify()

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
