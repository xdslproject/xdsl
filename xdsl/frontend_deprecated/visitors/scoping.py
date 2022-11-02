import ast
import logging

from dataclasses import dataclass
from typing import Any, Dict, Optional

import xdsl.ir
import xdsl.dialects.builtin
from xdsl.frontend_deprecated.block import block
from xdsl.frontend_deprecated.visitors.state import ProgramState
from xdsl.frontend_deprecated.visitors.utils import is_region, is_module


@dataclass
class ScopingVisitor(ast.NodeVisitor):
    """Traverses the Python AST and identifies and gathers labels of blocks."""

    def __init__(self, glob: Dict[str, Any], logger: Optional[logging.RootLogger] = None) -> None:
        if not logger:
            logger = logging.getLogger("scoping_visitor_logger")
            logger.setLevel(logging.INFO)
        self.logger = logger

        self.state = ProgramState(logger=self.logger)
        self.glob = glob

        super().__init__()

    # helper

    def _resolve(self, node: ast.AST) -> Any:
        """
        Resolve a node in the context (globals()) of the frontend program.
        """
        return self.glob.get(ast.unparse(node), None)

    def _add_module(self):
        self.state.enter_new_module(xdsl.dialects.builtin.ModuleOp([], [], []))

    def _add_region(self):
        if not self.state.has_current_module():
            self._add_module()
        self.state.enter_new_region(xdsl.ir.Region())

    def _add_op_region(self):
        if not self.state.is_in_block():
            self._add_block()
        self._add_region()

    def _add_block(self, label: Optional[str] = None):
        if not self.state.has_current_region():
            self._add_region()
        self.state.enter_new_block(xdsl.ir.Block(), label)

    # visit_*

    def visit_For(self, node: ast.For) -> None:
        self._add_op_region()
        self._add_block()

        for stmt in node.body:
            self.visit(stmt)

        self.state.exit_block()
        self.state.exit_region()

    def visit_FunctionDef(self, funcDef: ast.FunctionDef) -> None:
        decorators = list(map(self._resolve, funcDef.decorator_list))

        if block in decorators:
            self._add_block(funcDef.name)
        else:
            self._add_op_region()
            self._add_block()

        for stmt in funcDef.body:
            self.visit(stmt)

        self.state.exit_block()
        if not block in decorators:
            self.state.exit_region()

    def visit_With(self, node: ast.With) -> None:
        if is_region(node):
            self._add_region()
            for stmt in node.body:
                self.visit(stmt)
            self.state.exit_region()
        elif is_module(node):
            self._add_module()
            for stmt in node.body:
                self.visit(stmt)
            self.state.exit_module()
