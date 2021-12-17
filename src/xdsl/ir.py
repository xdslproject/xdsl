from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, TYPE_CHECKING, TypeVar
import typing

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


@dataclass
class SSAValue(ABC):
    """A reference to an SSA variable.
    An SSA variable is either an operation result, or a basic block argument."""

    typ: Attribute
    """Each SSA variable is associated to a type."""


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


@dataclass(frozen=True)
class Attribute(ABC):
    """
    A compile-time value.
    Attributes are used to represent SSA variable types, and can be attached
    on operations to give extra information.
    """

    name: str = field(default="", init=False)
    """The attribute name should be a static field in the attribute classes."""


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

    operands: List[SSAValue] = field(default_factory=list)
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

    def add_region(self, region: Region) -> None:
        self.regions.append(region)
        region.parent = self

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

    def __eq__(self, other: Operation) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(eq=False)
class Block:
    """A sequence of operations"""

    args: List[BlockArgument] = field(default_factory=list)
    """The basic block arguments."""

    ops: List[Operation] = field(default_factory=list)
    """Ordered operations contained in the block."""

    parent: Optional[Region] = field(default=None)
    """Parent region containing the block."""
    @staticmethod
    def with_arg_types(arg_types: List[Attribute]) -> Block:
        b = Block()
        b.args = [
            BlockArgument(typ, b, index) for index, typ in enumerate(arg_types)
        ]
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

    def __eq__(self, other: Block) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class Region:
    """A region contains a CFG of blocks. Regions are contained in operations."""

    blocks: List[Block] = field(default_factory=list)
    """Blocks contained in the region. The first block is the entry block."""

    parent: Optional[Operation] = field(default=None)
    """Operation containing the region."""
    @staticmethod
    def from_operation_list(ops: List[Operation]) -> Region:
        block = Block([], ops)
        region = Region()
        region.add_block(block)
        return region

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
