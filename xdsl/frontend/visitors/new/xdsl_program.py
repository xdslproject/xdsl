from dataclasses import dataclass, field
from typing import List, Optional
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue


@dataclass
class ProgramException(Exception):
    """
    Exception type if there was an error while construction the program.
    """
    msg: str

    def __str__(self) -> str:
        return f"Error while building the program: {self.msg}."


@dataclass
class XDSLProgram:
    """
    This class represents an xDSL program which AST visitor builds while parsing
    the Python AST.
    """

    modules: List[ModuleOp] = field(default_factory=list)
    """Modules defined in the current program."""

    insertion_point: Block | None = field(default=None)
    """Block into which we insert operations."""

    inferred_type: Attribute | None = field(default=None)

    stack: list[SSAValue] = field(default_factory=list)
    """Stack of operations."""

    def insert_op(self, op: Operation):
        """Inserts an operation."""
        if self.insertion_point is None:
            # This is a top-level operation, i.e. module! No insertion needed.
            self.modules.append(op)
            self.insertion_point = op.regions[-1].blocks[-1]
        else:
            self.insertion_point.add_op(op)
            if len(op.results) > 1:
                raise ProgramException(f"expected {op} to return a single result, but got {len(op.results)}")
            for result in op.results:
                self.stack.append(result)

    def insertion_point_from_op(self, op: Operation | None):
        # Special case: no top-level operation means it is a module.
        if op is None:
            self.insertion_point = None
            return

        if len(op.regions) == 0:
            raise ProgramException(f"{op} does not have regions")
        if len(op.regions[-1].blocks) == 0:
            raise ProgramException(f"{op} does not have blocks")
        self.insertion_point = op.regions[-1].blocks[-1]
    
    def insertion_point_from_region(self, region: Region):
        if len(region.blocks) == 0:
            raise ProgramException(f"{region} does not have blocks")
        self.insertion_point = region.blocks[-1]

    def insertion_point_from_block(self, block: Block):
        self.insertion_point = block
