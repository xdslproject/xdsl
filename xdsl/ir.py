from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from frozenlist import FrozenList
from io import StringIO
from typing import (TYPE_CHECKING, Any, Callable, Generic, Optional, Protocol,
                    Sequence, TypeVar, cast, Iterator, ClassVar)
import sys

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.parser import BaseParser
    from xdsl.printer import Printer
    from xdsl.irdl import OpDef, ParamAttrDef

OpT = TypeVar('OpT', bound='Operation')


@dataclass
class Dialect:
    """Contains the operations and attributes of a specific dialect"""
    _operations: list[type[Operation]] = field(default_factory=list,
                                               init=True,
                                               repr=True)
    _attributes: list[type[Attribute]] = field(default_factory=list,
                                               init=True,
                                               repr=True)

    @property
    def operations(self) -> Iterator[type[Operation]]:
        return iter(self._operations)

    @property
    def attributes(self) -> Iterator[type[Attribute]]:
        return iter(self._attributes)

    def __call__(self, ctx: MLContext) -> None:
        print(
            "Calling a dialect in order to register it is deprecated "
            "and will soon be removed.",
            file=sys.stderr)
        # TODO; Remove this function in a future release.
        assert isinstance(ctx, MLContext)
        ctx.register_dialect(self)


@dataclass
class MLContext:
    """Contains structures for operations/attributes registration."""
    _registeredOps: dict[str, type[Operation]] = field(default_factory=dict)
    _registeredAttrs: dict[str, type[Attribute]] = field(default_factory=dict)
    registered_unregistered_ops: dict[str, type[Operation]] = field(
        default_factory=dict)

    def register_dialect(self, dialect: Dialect):
        """Register a dialect. Operation and Attribute names should be unique"""
        for op in dialect.operations:
            self.register_op(op)

        for attr in dialect.attributes:
            self.register_attr(attr)

    def register_op(self, op: type[Operation]) -> None:
        """Register an operation definition. Operation names should be unique."""
        if op.name in self._registeredOps:
            raise Exception(f"Operation {op.name} has already been registered")
        self._registeredOps[op.name] = op

    def register_attr(self, attr: type[Attribute]) -> None:
        """Register an attribute definition. Attribute names should be unique."""
        if attr.name in self._registeredAttrs:
            raise Exception(
                f"Attribute {attr.name} has already been registered")
        self._registeredAttrs[attr.name] = attr

    def get_optional_op(self, name: str) -> type[Operation] | None:
        """Get an operation class from its name if it exists."""
        if name not in self._registeredOps:
            return None
        return self._registeredOps[name]

    def get_op(self, name: str) -> type[Operation]:
        """Get an operation class from its name."""
        if op_type := self.get_optional_op(name):
            return op_type
        raise Exception(f"Operation {name} is not registered")

    def get_optional_attr(self, name: str) -> type[Attribute] | None:
        """Get an attribute class from its name if it exists."""
        if name not in self._registeredAttrs:
            return None
        return self._registeredAttrs[name]

    def get_attr(self, name: str) -> type[Attribute]:
        """Get an attribute class from its name."""
        if attr_type := self.get_optional_attr(name):
            return attr_type
        raise Exception(f"Attribute {name} is not registered")


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

    uses: set[Use] = field(init=False, default_factory=set, repr=False)
    """All uses of the value."""

    _name: str | None = field(init=False, default=None)

    _name_regex: ClassVar[re.Pattern] = re.compile(
        r'[A-Za-z0-9._$-]*[A-Za-z._$-]')

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str):
        if self._name_regex.fullmatch(name):
            self._name = name

    @staticmethod
    def get(arg: SSAValue | Operation) -> SSAValue:
        "Get a new SSAValue from either a SSAValue, or an operation with a single result."
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
                "Attempting to delete SSA value that still has uses of operation:\n"
                f"{self.op}")
        self.replace_by(ErasedSSAValue(self.typ, self))


@dataclass
class OpResult(SSAValue):
    """A reference to an SSA variable defined by an operation result."""

    op: Operation
    """The operation defining the variable."""

    result_index: int
    """The index of the result in the defining operation."""

    def __repr__(self) -> str:
        return f"OpResult(typ={repr(self.typ)}, num_uses={repr(len(self.uses))}, " + \
               f"op_name={repr(self.op.name)}, " + \
               f"result_index={repr(self.result_index)}, name={repr(self.name)})"

    def __eq__(self, other: object) -> bool:
        return self is other

    # This might be problematic, as the superclass is not hashable ...
    def __hash__(self) -> int:  # type: ignore
        return id(self)


@dataclass
class BlockArgument(SSAValue):
    """A reference to an SSA variable defined by a basic block argument."""

    block: Block
    """The block defining the variable."""

    index: int
    """The index of the variable in the block arguments."""

    def __repr__(self) -> str:
        if isinstance(self.block, Block):
            block_repr = f"Block(num_arguments={len(self.block.args)}, " + \
                         f"num_blocks={len(self.block.ops)} ops)"
        else:
            block_repr = repr(self.block)
        return f"OpResult(typ={repr(self.typ)}, num_uses={repr(len(self.uses))}" + \
               f", block={block_repr}, index={repr(self.index)}"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:  # type: ignore
        return id(self)


@dataclass
class ErasedSSAValue(SSAValue):
    """
    An erased SSA variable.
    This is used during transformations when a SSA variable is destroyed but still used.
    """

    old_value: SSAValue

    def __hash__(self) -> int:  # type: ignore
        return hash(id(self))


@dataclass
class MLIRType:
    """
    A class representing an MLIR type.
    This class should only be inherited by classes inheriting Attribute.
    This class is only used for printing attributes in the MLIR format,
    inheriting this class prefix the attribute by `!` instead of `#`.
    """

    def __post_init__(self):
        if not isinstance(self, Attribute):
            raise TypeError(
                "MLIRType should only be inherited by classes inheriting Attribute"
            )


A = TypeVar('A', bound='Attribute')


class Attribute(ABC):
    """
    A compile-time value.
    Attributes are used to represent SSA variable types, and can be attached
    on operations to give extra information.
    """
    name: str = field(default="", init=False)
    """The attribute name should be a static field in the attribute classes."""

    @classmethod
    def build(cls: type[A], *args: Any) -> A:
        """Create a new attribute using one of the builder defined in IRDL."""
        assert False

    def __post_init__(self):
        self.verify()

    def verify(self) -> None:
        """
        Check that the attribute parameters satisfy the expected invariants.
        Raise an exception otherwise.
        """
        pass

    def __str__(self) -> str:
        from xdsl.printer import Printer
        res = StringIO()
        printer = Printer(stream=res)
        printer.print_attribute(self)
        return res.getvalue()


DataElement = TypeVar("DataElement")


@dataclass(frozen=True)
class Data(Generic[DataElement], Attribute, ABC):
    """An attribute represented by a Python structure."""
    data: DataElement

    @staticmethod
    @abstractmethod
    def parse_parameter(parser: BaseParser) -> DataElement:
        """Parse the attribute parameter."""

    @staticmethod
    @abstractmethod
    def print_parameter(data: DataElement, printer: Printer) -> None:
        """Print the attribute parameter."""


@dataclass(frozen=True)
class ParametrizedAttribute(Attribute):
    """An attribute parametrized by other attributes."""
    parameters: list[Attribute] = field(default_factory=list)

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        """Parse the attribute parameters."""
        return parser.parse_paramattr_parameters()

    def print_parameters(self, printer: Printer) -> None:
        """Print the attribute parameters."""
        printer.print_paramattr_parameters(self.parameters)

    @classmethod
    @property
    def irdl_definition(cls) -> ParamAttrDef:
        """Get the IRDL attribute definition."""
        ...


@dataclass
class IRNode(ABC):

    parent: IRNode | None

    def is_ancestor(self, op: IRNode) -> bool:
        "Returns true if the IRNode is an ancestor of another IRNode."
        if op is self:
            return True
        if op.parent is None:
            return False
        return self.is_ancestor(op.parent)

    def get_toplevel_object(self) -> IRNode:
        """Get the operation, block, or region ancestor that has no parents."""
        if self.parent is None:
            return self
        return self.parent.get_toplevel_object()

    def is_structurally_equivalent(
            self,
            other: IRNode,
            context: Optional[dict[IRNode, IRNode]] = None) -> bool:
        """Check if two IR nodes are structurally equivalent."""
        ...


@dataclass
class Operation(IRNode):
    """A generic operation. Operation definitions inherit this class."""

    name: str = field(default="", init=False)
    """The operation name. Should be a static member of the class"""

    _operands: FrozenList[SSAValue] = field(default_factory=FrozenList)
    """The operation operands."""

    results: list[OpResult] = field(default_factory=list)
    """The results created by the operation."""

    successors: list[Block] = field(default_factory=list)
    """
    The basic blocks that the operation may give control to.
    This list should be empty for non-terminator operations.
    """

    attributes: dict[str, Attribute] = field(default_factory=dict)
    """The attributes attached to the operation."""

    regions: list[Region] = field(default_factory=list)
    """Regions arguments of the operation."""

    parent: Block | None = field(default=None, repr=False)
    """The block containing this operation."""

    def parent_op(self) -> Operation | None:
        if p := self.parent_region():
            return p.parent
        return None

    def parent_region(self) -> Region | None:
        if p := self.parent_block():
            return p.parent
        return None

    def parent_block(self) -> Block | None:
        return self.parent

    @property
    def operands(self) -> FrozenList[SSAValue]:
        return self._operands

    @operands.setter
    def operands(self, new: list[SSAValue] | FrozenList[SSAValue]):
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
    def with_result_types(
            op: Any,
            operands: Sequence[SSAValue] | None = None,
            result_types: Sequence[Attribute] | None = None,
            attributes: dict[str, Attribute] | None = None,
            successors: Sequence[Block] | None = None,
            regions: Sequence[Region] | None = None) -> Operation:

        operation = op()
        if operands is not None:
            operation.operands = operands
        if result_types:
            operation.results = [
                OpResult(typ, operation, idx)
                for (idx, typ) in enumerate(result_types)
            ]
        if attributes:
            operation.attributes = attributes
        if successors:
            operation.successors = successors
        if regions:
            for region in regions:
                operation.add_region(region)
        return operation

    @classmethod
    def create(cls: type[OpT],
               operands: Sequence[SSAValue] | None = None,
               result_types: Sequence[Attribute] | None = None,
               attributes: dict[str, Attribute] | None = None,
               successors: Sequence[Block] | None = None,
               regions: Sequence[Region] | None = None) -> OpT:
        op = Operation.with_result_types(cls, operands, result_types,
                                         attributes, successors, regions)
        return cast(OpT, op)

    @classmethod
    def build(cls: type[OpT],
              operands: list[Any] | None = None,
              result_types: list[Any] | None = None,
              attributes: dict[str, Any] | None = None,
              successors: list[Any] | None = None,
              regions: list[Any] | None = None) -> OpT:
        """Create a new operation using builders."""
        ...

    def replace_operand(self, operand_idx: int, new_operand: SSAValue) -> None:
        """Replace an operand with another operand."""
        self.operands = list(self._operands[:operand_idx]) + [
            new_operand
        ] + list(self._operands[operand_idx + 1:])

    def add_region(self, region: Region) -> None:
        """Add an unattached region to the operation."""
        if region.parent:
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
        """
        Call a function on all operations contained in the operation (including this one)
        """
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

    _OperationType = TypeVar('_OperationType', bound='Operation')

    @classmethod
    def parse(cls: type[_OperationType], result_types: list[Attribute],
              parser: BaseParser) -> _OperationType:
        return parser.parse_op_with_default_format(cls, result_types)

    def print(self, printer: Printer):
        return printer.print_op_with_default_format(self)

    def clone_without_regions(
            self: OpT,
            value_mapper: dict[SSAValue, SSAValue] | None = None,
            block_mapper: dict[Block, Block] | None = None) -> OpT:
        """Clone an operation, with empty regions instead."""
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}
        operands = [
            (value_mapper[operand] if operand in value_mapper else operand)
            for operand in self.operands
        ]
        result_types = [res.typ for res in self.results]
        attributes = self.attributes.copy()
        successors = [(block_mapper[successor]
                       if successor in block_mapper else successor)
                      for successor in self.successors]
        regions = [Region() for _ in self.regions]
        cloned_op = self.create(operands=operands,
                                result_types=result_types,
                                attributes=attributes,
                                successors=successors,
                                regions=regions)
        for idx, result in enumerate(cloned_op.results):
            value_mapper[self.results[idx]] = result
        return cloned_op

    def clone(self: OpT,
              value_mapper: dict[SSAValue, SSAValue] | None = None,
              block_mapper: dict[Block, Block] | None = None) -> OpT:
        """Clone an operation with all its regions and operations in them."""
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}
        op = self.clone_without_regions(value_mapper, block_mapper)
        for idx, region in enumerate(self.regions):
            region.clone_into(op.regions[idx], 0, value_mapper, block_mapper)
        return op

    def erase(self,
              safe_erase: bool = True,
              drop_references: bool = True) -> None:
        """
        Erase the operation, and remove all its references to other operations.
        If safe_erase is specified, check that the operation results are not used.
        """
        assert self.parent is None, "Operation with parents should first be detached " + \
                                    "before erasure."
        if drop_references:
            self.drop_all_references()
        for result in self.results:
            result.erase(safe_erase=safe_erase)

    def detach(self):
        """Detach the operation from its parent block."""
        if self.parent is None:
            raise Exception("Cannot detach a toplevel operation.")
        self.parent.detach_op(self)

    def is_structurally_equivalent(
            self,
            other: IRNode,
            context: Optional[dict[IRNode, IRNode]] = None) -> bool:
        """
        Check if two operations are structurally equivalent.
        The context is a mapping of IR nodes to IR nodes that are already known "
        "to be equivalent. This enables checking whether the use dependencies and "
        "successors are equivalent.
        """
        if context is None:
            context = {}
        if not isinstance(other, Operation):
            return False
        if self.name != other.name:
            return False
        if len(self.operands) != len(other.operands) or \
           len(self.results) != len(other.results) or \
           len(self.regions) != len(other.regions) or \
           len(self.successors) != len(other.successors) or \
           self.attributes != other.attributes:
            return False
        if self.parent and other.parent and context.get(
                self.parent) != other.parent:
            return False
        if not all(
                context.get(operand) == other_operand for operand,
                other_operand in zip(self.operands, other.operands)):
            return False
        if not all(
                context.get(successor) == other_successor for successor,
                other_successor in zip(self.successors, other.successors)):
            return False
        if not all(
                region.is_structurally_equivalent(other_region, context)
                for region, other_region in zip(self.regions, other.regions)):
            return False
        # Add results of this operation to the context
        for result, other_result in zip(self.results, other.results):
            context[result] = other_result

        return True

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    @classmethod
    @property
    def irdl_definition(cls) -> OpDef:
        """Get the IRDL operation definition."""
        ...


@dataclass()
class Block(IRNode):
    """A sequence of operations"""

    declared_at: 'Span' | None = None

    _args: FrozenList[BlockArgument] = field(default_factory=FrozenList,
                                             init=False)
    """The basic block arguments."""

    ops: list[Operation] = field(default_factory=list, init=False)
    """Ordered operations contained in the block."""

    parent: Region | None = field(default=None, init=False, repr=False)
    """Parent region containing the block."""

    def parent_op(self) -> Operation | None:
        return self.parent.parent if self.parent else None

    def parent_region(self) -> Region | None:
        return self.parent

    def parent_block(self) -> Block | None:
        return self.parent.parent.parent if self.parent and self.parent.parent else None

    def __repr__(self) -> str:
        return f"Block(_args={repr(self._args)}, num_ops={len(self.ops)})"

    @property
    def args(self) -> FrozenList[BlockArgument]:
        """Returns the block arguments."""
        return self._args

    @staticmethod
    def from_arg_types(arg_types: list[Attribute]) -> Block:
        b = Block()
        b._args = FrozenList([
            BlockArgument(typ, b, index) for index, typ in enumerate(arg_types)
        ])
        b._args.freeze()
        return b

    @staticmethod
    def from_ops(ops: list[Operation],
                 arg_types: list[Attribute] | None = None):
        b = Block()
        if arg_types:
            b._args = FrozenList([
                BlockArgument(typ, b, index)
                for index, typ in enumerate(arg_types)
            ])
            b._args.freeze()
        b.add_ops(ops)
        return b

    class BlockCallback(Protocol):

        def __call__(self, *args: BlockArgument) -> list[Operation]:
            ...

    @staticmethod
    def from_callable(block_arg_types: list[Attribute], f: BlockCallback):
        b = Block.from_arg_types(block_arg_types)
        b.add_ops(f(*b.args))
        return b

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
        if operation.parent:
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

    def add_ops(self, ops: list[Operation]) -> None:
        """
        Add operations at the end of the block.
        The operations should not be attached to another block.
        """
        for op in ops:
            self.add_op(op)

    def insert_op(self,
                  ops: Operation | list[Operation],
                  index: int,
                  name: str | None = None) -> None:
        """
        Insert one or multiple operations at a given index in the block.
        The operations should not be attached to another block.
        """
        if index < 0 or index > len(self.ops):
            raise ValueError(
                f"Can't insert operation in index {index} in a block with "
                f"{len(self.ops)} operations.")
        if not isinstance(ops, list):
            ops = [ops]
        if name:
            for curr_op in ops:
                for res in curr_op.results:
                    res.name = name
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

    def detach_op(self, op: int | Operation) -> Operation:
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

    def erase_op(self, op: int | Operation, safe_erase: bool = True) -> None:
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

    def erase(self, safe_erase: bool = True) -> None:
        """
        Erase the block, and remove all its references to other operations.
        If safe_erase is specified, check that no operation results are used outside
        the block.
        """
        assert self.parent is None, "Blocks with parents should first be detached " + \
                                    "before erasure."
        self.drop_all_references()
        for op in self.ops:
            op.erase(safe_erase=safe_erase, drop_references=False)

    def is_structurally_equivalent(
            self,
            other: IRNode,
            context: Optional[dict[IRNode, IRNode]] = None) -> bool:
        """
        Check if two blocks are structurally equivalent.
        The context is a mapping of IR nodes to IR nodes that are already known "
        "to be equivalent. This enables checking whether the use dependencies and "
        "successors are equivalent.
        """
        if context is None:
            context = {}
        if not isinstance(other, Block):
            return False
        if len(self.args) != len(other.args) or \
           len(self.ops) != len(other.ops):
            return False
        for arg, other_arg in zip(self.args, other.args):
            if arg.typ != other_arg.typ:
                return False
            context[arg] = other_arg
        # Add self to the context so Operations can check for identical parents
        context[self] = other
        if not all(
                op.is_structurally_equivalent(other_op, context)
                for op, other_op in zip(self.ops, other.ops)):
            return False

        return True

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class Region(IRNode):
    """A region contains a CFG of blocks. Regions are contained in operations."""

    blocks: list[Block] = field(default_factory=list, init=False)
    """Blocks contained in the region. The first block is the entry block."""

    parent: Operation | None = field(default=None, init=False, repr=False)
    """Operation containing the region."""

    def parent_block(self) -> Block | None:
        return self.parent.parent if self.parent else None

    def parent_op(self) -> Operation | None:
        return self.parent

    def parent_region(self) -> Region | None:
        return self.parent.parent.parent if self.parent and self.parent.parent else None

    def __repr__(self) -> str:
        return f"Region(num_blocks={len(self.blocks)})"

    @staticmethod
    def from_operation_list(ops: list[Operation]) -> Region:
        block = Block.from_ops(ops)
        region = Region()
        region.add_block(block)
        return region

    @staticmethod
    def from_block_list(blocks: list[Block]) -> Region:
        region = Region()
        for block in blocks:
            region.add_block(block)
        return region

    @staticmethod
    def get(arg: Region | list[Block] | list[Operation]) -> Region:
        if isinstance(arg, Region):
            return arg
        if isinstance(arg, list):
            if len(arg) == 0:
                return Region.from_operation_list([])
            if isinstance(arg[0], Block):
                return Region.from_block_list(cast(list[Block], arg))
            if isinstance(arg[0], Operation):
                return Region.from_operation_list(cast(list[Operation], arg))
        raise TypeError(f"Can't build a region with argument {arg}")

    @property
    def ops(self) -> list[Operation]:
        """
        Get the operations of a single-block region.
        Returns an exception if the region is not single-block.
        """
        if len(self.blocks) != 1:
            raise ValueError(
                "'ops' property of Region class is only available "
                "for single-block regions.")
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
        if block.parent:
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

    def insert_block(self, blocks: Block | list[Block], index: int) -> None:
        """
        Insert one or multiple blocks at a given index in the region.
        The blocks should not be attached to another region.
        """
        if index < 0 or index > len(self.blocks):
            raise ValueError(
                f"Can't insert block in index {index} in a block with "
                f"{len(self.blocks)} blocks.")
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

    def detach_block(self, block: int | Block) -> Block:
        """
        Detach a block from the region.
        Returns the detached block.
        """
        if isinstance(block, Block):
            block_idx = self.get_block_index(block)
        else:
            block_idx = block
            block = self.blocks[block_idx]
        block.parent = None
        self.blocks = self.blocks[:block_idx] + self.blocks[block_idx + 1:]
        return block

    def erase_block(self, block: int | Block, safe_erase: bool = True) -> None:
        """
        Erase a block from the region.
        If safe_erase is True, check that the block has no uses.
        """
        block = self.detach_block(block)
        block.erase(safe_erase=safe_erase)

    def clone_into(self,
                   dest: Region,
                   insert_index: int | None = None,
                   value_mapper: dict[SSAValue, SSAValue] | None = None,
                   block_mapper: dict[Block, Block] | None = None):
        """
        Clone all block of this region into `dest` to position `insert_index`
        """
        assert dest and dest != self
        if insert_index is None:
            insert_index = len(dest.blocks)
        if value_mapper is None:
            value_mapper = {}
        if block_mapper is None:
            block_mapper = {}

        for block in self.blocks:
            new_block = Block()
            block_mapper[block] = new_block
            for idx, block_arg in enumerate(block.args):
                new_block.insert_arg(block_arg.typ, idx)
                value_mapper[block_arg] = new_block.args[idx]
            for op in block.ops:
                new_block.add_op(op.clone(value_mapper, block_mapper))
            dest.insert_block(new_block, insert_index)
            insert_index += 1

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
        assert self.parent, "Regions with parents should first be " + \
                            "detached before erasure."
        self.drop_all_references()

    def move_blocks(self, region: Region) -> None:
        """
        Move the blocks of this region to another region. Leave no blocks in this region.
        """
        region.blocks = self.blocks
        self.blocks = []
        for block in region.blocks:
            block.parent = region

    def is_structurally_equivalent(
            self,
            other: IRNode,
            context: Optional[dict[IRNode, IRNode]] = None) -> bool:
        """
        Check if two regions are structurally equivalent.
        The context is a mapping of IR nodes to IR nodes that are already known "
        "to be equivalent. This enables checking whether the use dependencies and "
        "successors are equivalent.
        """
        if context is None:
            context = {}
        if not isinstance(other, Region):
            return False
        if len(self.blocks) != len(other.blocks):
            return False
        # register all blocks in the context so we can check whether ops have
        # the corrects successors
        for block, other_block in zip(self.blocks, other.blocks):
            context[block] = other_block
        if not all(
                block.is_structurally_equivalent(other_block, context)
                for block, other_block in zip(self.blocks, other.blocks)):
            return False
        return True
