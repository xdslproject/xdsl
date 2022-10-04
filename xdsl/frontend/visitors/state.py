import re
import logging
from dataclasses import dataclass, field
from importlib.metadata import metadata
from typing import List, Optional, Tuple, Union, Dict

import xdsl.ir
import xdsl.dialects.builtin
from xdsl.frontend.block import block


class StateException(Exception):
    def __str__(self) -> str:
        return f"Exception in the internal program state: {super().__str__()}"


_LOOKUP_VARIABLE_RETURN_TYPE = Tuple[Optional[xdsl.ir.Attribute],
                                     Optional[xdsl.ir.BlockArgument]]


def _lookup_variable_in_list(name: str,
                             l: List[Union['BlockMetadata',
                                           'RegionMetadata', 'ModuleMetadata']]
                             ) -> _LOOKUP_VARIABLE_RETURN_TYPE:
    for md in l:
        t = md.lookup_variable(name)
        if t:
            return t
    return None, None


@dataclass
class BlockMetadata:
    block: xdsl.ir.Block
    label: str

    # Blocks can have regions that are attached to operations. We track them here
    # without the mapping to which operation they are attached because it is not
    # relevant for the information about the scope.
    op_region_metadatas: List['RegionMetadata'] = field(default_factory=list)

    curr_op_region_idx: int = -1

    vars: Dict[str, xdsl.ir.Attribute] = field(default_factory=dict)

    def add(self, region: xdsl.ir.Region):
        """
        Add another operand region (for a block with regions) to the current block.
        Switch execution to that new region.
        """
        block_metadata = [BlockMetadata(
            block, f"bb{i}") for i, block in enumerate(region.blocks)]
        self.op_region_metadatas.append(RegionMetadata(region, block_metadata))
        self.curr_op_region_idx = len(self.op_region_metadatas) - 1
        return self.curr_op_region_idx

    def get_next(self) -> Tuple[int, xdsl.ir.Region]:
        next_idx = self.curr_op_region_idx + 1

        if next_idx >= len(self.op_region_metadatas):
            raise StateException("no next op region available to enter.")

        self.curr_op_region_idx = next_idx
        return next_idx, self.op_region_metadatas[next_idx].region

    def reset(self) -> None:
        self.curr_op_region_idx = -1
        for op_region_metadata in self.op_region_metadatas:
            op_region_metadata.reset()

    def add_var(self, name: str, var_type: xdsl.ir.Attribute, block_arg: Optional[xdsl.ir.BlockArgument]) -> None:
        self.vars[name] = (var_type, block_arg)

    def lookup_variable(self, name: str) -> _LOOKUP_VARIABLE_RETURN_TYPE:
        if name in self.vars:
            return self.vars[name]
        return _lookup_variable_in_list(name, self.op_region_metadatas)


@dataclass
class RegionMetadata:
    region: xdsl.ir.Region = field(default_factory=xdsl.ir.Region)
    block_metadatas: List[BlockMetadata] = field(default_factory=list)
    curr_block_idx: int = -1

    def add(self, block: xdsl.ir.Block, label: str):
        """
        Add new block to the current region and make it the currently active block.
        """
        self.region.add_block(block)
        self.block_metadatas.append(BlockMetadata(block, label))
        self.curr_block_idx = len(self.block_metadatas) - 1
        return self.curr_block_idx

    def _get_next_idx_by_label(self, label: str) -> int:
        for idx, block_metadata in enumerate(self.block_metadatas):
            if block_metadata.label == label:
                return idx
        raise StateException(f"No block with label {label} in current scope.")

    def get_next(self, label: Optional[str] = None) -> Tuple[int, xdsl.ir.Block]:
        if label:
            next_idx = self._get_next_idx_by_label(label)
        else:
            next_idx = self.curr_block_idx + 1
            if next_idx >= len(self.block_metadatas):
                raise StateException("no next block available to enter.")

        self.curr_block_idx = next_idx
        return next_idx, self.block_metadatas[next_idx].block

    def reset(self) -> None:
        self.curr_block_idx = -1
        for block_metadata in self.block_metadatas:
            block_metadata.reset()

    def lookup_variable(self, name: str) -> _LOOKUP_VARIABLE_RETURN_TYPE:
        return _lookup_variable_in_list(name, self.block_metadatas)


@dataclass
class ModuleMetadata:
    module: xdsl.dialects.builtin.ModuleOp
    region_metadatas: List[RegionMetadata] = field(default_factory=list)
    curr_region_idx: int = -1

    def add(self, region: xdsl.ir.Region) -> int:
        """Add new region to the current module and make it the currently active one."""
        self.module.add_region(region)
        block_metadata = [BlockMetadata(
            block, f"bb{i}") for i, block in enumerate(region.blocks)]
        self.region_metadatas.append(RegionMetadata(region, block_metadata))
        self.curr_region_idx = len(self.region_metadatas) - 1
        return self.curr_region_idx

    def get_next(self) -> Tuple[int, xdsl.ir.Region]:
        next_idx = self.curr_region_idx + 1
        if next_idx >= len(self.region_metadatas):
            raise StateException("no next region available to enter.")

        self.curr_region_idx = next_idx
        return next_idx, self.region_metadatas[next_idx].region

    def reset(self) -> None:
        self.curr_region_idx = -1
        for region_metadata in self.region_metadatas:
            region_metadata.reset()

    def lookup_variable(self, name: str) -> _LOOKUP_VARIABLE_RETURN_TYPE:
        return _lookup_variable_in_list(name, self.region_metadatas)


class ProgramState:
    """
    This class tracks the scope of the program, which includes the hierarchy
    of modules, regions, and blocks in which the execution currently is.
    Moreover, it tracks the variables that are currently in scope.

    The data structure is intended to be append only. Added modules/regions/blocks 
    should not be removed.

    XXX: This class could probably be designed more nicely. But the need
        for both a hierarchical structure and a stack layout made it a bit difficult.
    """

    def __init__(self, logger: Optional[logging.RootLogger] = None) -> None:
        self.module_metadata_stack: List[ModuleMetadata] = []

        # current_indices is a stack holding the indices to the currently active block
        # It is used to track the (potentially nested) set of modules, regions, and blocks
        # that are currently in scope
        self.current_indices = []
        self.last_module_idx = -1

        if not logger:
            logger = logging.getLogger("program_state_logger")
            logger.setLevel(logging.INFO)
        self.logger = logger

    def get_current_module(self) -> ModuleMetadata:
        if len(self.current_indices) <= 0:
            raise StateException("no current scope defined.")
        return self.module_metadata_stack[self.current_indices[0]]

    def get_current_scope(self) -> BlockMetadata | RegionMetadata | ModuleMetadata:
        module = self.get_current_module()

        if len(self.current_indices) == 1:
            return module

        regions = module.region_metadatas
        for current_indices_idx in range(1, len(self.current_indices), 2):
            current_idx = self.current_indices[current_indices_idx]
            region = regions[current_idx]

            if current_indices_idx + 1 < len(self.current_indices):
                next_idx = current_idx = self.current_indices[current_indices_idx + 1]
                block = region.block_metadatas[next_idx]
                regions = block.op_region_metadatas
            else:
                return region
        return block

    # create new objects
    def enter_new_module(self, module: xdsl.dialects.builtin.ModuleOp) -> None:
        self.logger.debug("Scoping: create new module and enter it")
        self.current_indices = [len(self.module_metadata_stack)]
        self.module_metadata_stack.append(
            ModuleMetadata(module))

    def enter_new_region(self, region: xdsl.ir.Region) -> None:
        metadata = self.get_current_scope()

        if isinstance(metadata, RegionMetadata):
            raise StateException("cannot nest regions directly. Nested regions need to be "
                                 "attached to operations.")

        self.logger.debug("Scoping: create new region and enter it")
        new_idx = metadata.add(region)
        self.current_indices.append(new_idx)

    def enter_new_op_region(self, region: xdsl.ir.Region) -> None:
        return self.enter_new_region(region, True)

    def enter_new_block(self, block: xdsl.ir.Block, label: Optional[str] = None) -> None:
        metadata = self.get_current_scope()

        if not isinstance(metadata, RegionMetadata):
            raise StateException(
                "blocks can only be attached to regions, not blocks or modules.")

        if not label:
            label_matches = [re.match(r'^bb(\d+)$', md.label)
                             for md in metadata.block_metadatas]
            if label_matches:
                idx = max([int(m.groups()[0]) for m in label_matches]) + 1
            else:
                idx = 0
            label = f"bb{idx}"

        self.logger.debug(f"Scoping: create new block '{label}' and enter it")
        new_idx = metadata.add(block, label)
        self.current_indices.append(new_idx)

    # check
    def has_current_module(self) -> bool:
        """Returns true when the current scope hierarchy has a module on top level."""
        return len(self.current_indices) > 0

    def has_current_region(self) -> bool:
        """Returns true when the current scope hierarchy has a valid region on the second level."""
        return len(self.current_indices) > 1 \
            and self.current_indices[1] < len(self.module_metadata_stack[self.current_indices[0]].region_metadatas)

    def has_block_with_label(self, label: str) -> bool:
        """Returns true when the current scope has a block of the given label."""
        if len(self.current_indices) > 0:
            metadata = self.get_current_scope()
            if isinstance(metadata, RegionMetadata):
                for block_metadata in metadata.block_metadatas:
                    if block_metadata.label == label:
                        return True
            return False

    def is_in_module(self) -> bool:
        """Returns true when the current scope is a module."""
        if len(self.current_indices) == 0:
            return False
        return isinstance(self.get_current_scope(), ModuleMetadata)

    def is_in_region(self) -> bool:
        """Returns true when the current scope is a region."""
        if len(self.current_indices) == 0:
            return False
        return isinstance(self.get_current_scope(), RegionMetadata)

    def is_in_block(self) -> bool:
        """Returns true when the current scope is a block."""
        if len(self.current_indices) == 0:
            return False
        return isinstance(self.get_current_scope(), BlockMetadata)

    # navigation
    def reset(self) -> None:
        """Reset the indices of the current program state (but keep the metadata information)."""
        for module_metadata in self.module_metadata_stack:
            module_metadata.reset()
        self.current_indices = []

    def enter_module(self) -> xdsl.dialects.builtin.ModuleOp:
        if len(self.current_indices) != 0:
            raise StateException("cannot nest modules. Exit the current module "
                                 "before entering a new one.")

        new_module_idx = self.last_module_idx + 1

        if new_module_idx >= len(self.module_metadata_stack):
            raise StateException("no new module to enter. (You might want to create"
                                 "a new one instead with `enter_new_module`).")

        self.logger.debug("Scoping: enter module")
        self.current_indices = [new_module_idx]
        return self.module_metadata_stack[new_module_idx].module

    def enter_region(self) -> xdsl.ir.Region:
        if len(self.current_indices) == 0:
            self.enter_module()

        metadata = self.get_current_scope()
        if isinstance(metadata, ModuleMetadata) or isinstance(metadata, BlockMetadata):
            self.logger.debug("Scoping: enter region")
            next_idx, next_region = metadata.get_next()
            self.current_indices.append(next_idx)
            return next_region

        raise StateException(
            f"Unknown state {type(metadata)}, expected block, module, or region.")

    def enter_op_region(self) -> xdsl.ir.Region:
        if not self.is_in_block():
            self.enter_block()
        return self.enter_region()

    def enter_block(self, label: Optional[str] = None) -> xdsl.ir.Block:
        if len(self.current_indices) == 0:
            # Add default module
            self.enter_module()

        metadata = self.get_current_scope()
        if not isinstance(metadata, RegionMetadata):
            # Add default region
            self.enter_region()
            metadata = self.get_current_scope()

        self.logger.debug(f"Scoping: enter block '{label}'")
        next_idx, next_block = metadata.get_next(label)
        self.current_indices.append(next_idx)

        return next_block

    def exit_module(self) -> None:
        metadata = self.get_current_scope()
        if not isinstance(metadata, ModuleMetadata):
            name = "region" if isinstance(
                metadata, RegionMetadata) else "block"
            raise StateException(f"Cannot exit module before exiting {name}.")

        self.logger.debug("Scoping: exit module")
        self.last_module_idx = self.current_indices[0]
        self.current_indices = []

    def exit_region(self) -> None:
        metadata = self.get_current_scope()
        if not isinstance(metadata, RegionMetadata):
            name = "block" if isinstance(metadata, BlockMetadata) else "module"
            raise StateException(f"Cannot exit region before exiting {name}.")

        self.logger.debug("Scoping: exit region")
        self.current_indices.pop()

    def exit_block(self) -> None:
        metadata = self.get_current_scope()
        if not isinstance(metadata, BlockMetadata):
            name = "region" if isinstance(
                metadata, RegionMetadata) else "module"
            raise StateException(f"Cannot exit block before exiting {name}.")

        self.logger.debug(f"Scoping: exit block '{metadata.label}'")
        self.current_indices.pop()

    # Ops
    def add_ops(self, ops: List[xdsl.ir.Operation]) -> None:
        if not self.is_in_block():
            raise StateException(
                "Cannot add operations when the current scope is not in a block.")

        metadata = self.get_current_scope()
        if hasattr(ops, "__iter__"):
            metadata.block.add_ops(ops)
        else:
            metadata.block.add_op(ops)

    # vars
    def add_variable(self, var_name: str,
                     var_type: xdsl.ir.Attribute,
                     block_arg: Optional[xdsl.ir.BlockArgument] = None) -> None:
        """Add a variable and its type to the block in the current scope.

        If the variable stands for a block argument, pass this as an optional argument.
        """
        self.logger.debug(
            f"Add new variable '{var_name}' of type '{var_type}' to state.")

        typ, _ = self.lookup_variable(var_name)
        if typ:
            raise StateException(
                f"Variable '{var_name}' already exists in state.")

        metadata = self.get_current_scope()

        if not isinstance(metadata, BlockMetadata):
            name = "region" if isinstance(
                metadata, RegionMetadata) else "module"
            raise StateException(f"cannot add variables to {name}s.")

        metadata.add_var(var_name, var_type, block_arg)

    def lookup_variable(self, var_name: str) -> xdsl.ir.Attribute:
        module = self.get_current_module()
        return module.lookup_variable(var_name)

    # Finalizers
    def finalize_region(self, stmts: List[xdsl.ir.Operation] | xdsl.ir.Block | xdsl.ir.Region) -> None:
        metadata = self.get_current_scope()
        if isinstance(stmts, list) and all([isinstance(e, xdsl.ir.Operation) for e in stmts]):
            if not isinstance(metadata, BlockMetadata):
                raise StateException("Trailing operations could not be added to block because "
                                     "there is no active block.")

            self.exit_block()
            self.exit_region()
        elif isinstance(stmts, xdsl.ir.Block):
            if not isinstance(metadata, RegionMetadata):
                raise StateException(
                    "Invalid state, expected scoping to be in region, not block.")
            self.exit_region()
        elif isinstance(stmts, xdsl.ir.Region) and not isinstance(metadata, ModuleMetadata):
            raise StateException(
                "Invalid state, expected scoping to be in module, not region.")
        else:
            raise StateException(
                f"Unexpected statements {stmts} of type {type(stmts)} in finalizer.")

    def finalize_module(self):
        if len(self.current_indices) == 1:
            self.exit_module()
        elif len(self.current_indices) > 1:
            raise StateException(
                "Not ready to close module, unclosed regions/blocks remain.")
        # else do nothing if we already exited the module (i.e., len = 0)
