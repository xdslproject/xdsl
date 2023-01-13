import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set
from xdsl.frontend.block import is_block
from xdsl.frontend.const import Const
from xdsl.frontend.exception import CodeGenerationException


@dataclass
class PythonCodeCheck:

    @staticmethod
    def run(stmts: List[ast.stmt]) -> None:
        """
        Checks if Python code within `CodeContext` is supported. On unsupported
        cases, an exception is raised. Particularly, the check is used for
        scoping because Python (as an interpreted language) does not have a
        notion of constants, global variables, etc. Hence, one can redefine
        functions, place variables in arbitrary locations and do many other
        weird things which xDSL/MLIR would not like.
        """

        # Any code written within code context can be organized either as a
        # sequence of instructions (interpreted one by one), for example:
        # ```
        # with CodeContext(p):
        #   a: i32 = 45
        #   b: i32 = 23
        #   result: i32 = a + b
        # ```
        # Alternatively, the code can be viewed as a sequence of functions
        # (possibly with a dedicated entry point), for example:
        # ```
        # with CodeContext(p):
        #  def foo(x: i32) -> i32:
        #    return x
        #  def main():
        #    y: i32 = foo(100)
        # ```
        # First, find out which of the two modes is used.
        single_scope: bool = True
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef) and not is_block(stmt):
                single_scope = False
                break

        # Check Python code is correctly structured.
        StructureCheck.run_with_scope(single_scope, stmts)

        # Check constant/global variables are correctly defined. Should be
        # called only after the structure is checked.
        ConstantCheck.run(stmts)


@dataclass
class StructureCheck:

    @staticmethod
    def run_with_scope(single_scope: bool, stmts: List[ast.stmt]) -> None:
        if single_scope:
            visitor = SingleScopeVisitor()
        else:
            visitor = MultipleScopeVisitor()
        for stmt in stmts:
            visitor.visit(stmt)


@dataclass
class SingleScopeVisitor(ast.NodeVisitor):

    block_names: Set[str] = field(default_factory=set)
    """Tracks duplicate block labels."""

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        assert is_block(node)

        if node.name in self.block_names:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Block '{node.name}' is already defined.")
        self.block_names.add(node.name)

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if is_block(stmt):
                    raise CodeGenerationException(
                        stmt.lineno, stmt.col_offset,
                        f"Cannot have a nested block '{stmt.name}' inside the block '{node.name}'."
                    )
                else:
                    raise CodeGenerationException(
                        stmt.lineno, stmt.col_offset,
                        f"Cannot have an inner function '{stmt.name}' inside the block '{node.name}'."
                    )


@dataclass
class MultipleScopeVisitor(ast.NodeVisitor):

    function_and_block_names: Dict[str, Set[str]] = field(default_factory=dict)
    """Tracks duplicate function names and duplicate block labels."""

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        assert not is_block(node)

        if node.name in self.function_and_block_names:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Function '{node.name}' is already defined.")
        self.function_and_block_names[node.name] = set()

        # Functions cannot have inner functions but can have blocks inside
        # which we still have to check.
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if not is_block(stmt):
                    raise CodeGenerationException(
                        stmt.lineno, stmt.col_offset,
                        f"Cannot have an inner function '{stmt.name}' inside the function '{node.name}'."
                    )
                else:
                    if stmt.name in self.function_and_block_names[node.name]:
                        raise CodeGenerationException(
                            stmt.lineno, stmt.col_offset,
                            f"Block '{stmt.name}' is already defined in function '{node.name}'."
                        )
                    self.function_and_block_names[node.name].add(stmt.name)

                    for inner_stmt in stmt.body:
                        if isinstance(inner_stmt, ast.FunctionDef):
                            if is_block(inner_stmt):
                                raise CodeGenerationException(
                                    inner_stmt.lineno, inner_stmt.col_offset,
                                    f"Cannot have a nested block '{inner_stmt.name}' inside the block '{stmt.name}'."
                                )
                            else:
                                raise CodeGenerationException(
                                    inner_stmt.lineno, inner_stmt.col_offset,
                                    f"Cannot have an inner function '{inner_stmt.name}' inside the block '{stmt.name}'."
                                )


@dataclass
class ConstantCheck:

    @staticmethod
    def run(stmts: List[ast.stmt]) -> None:
        visitor = ConstantVisitor()
        for stmt in stmts:
            visitor.visit(stmt)


@dataclass
class Constant:

    value: int | float
    """Evaluated value of the constant. Only a few primitive types are supported."""

    shadowed: bool = field(default=False)
    """True if this constant is shadowed by function/block argument."""


@dataclass
class ConstantVisitor(ast.NodeVisitor):

    constants: Dict[str, Constant] = field(default_factory=dict)
    """Stores all information about constants in the current system."""

    global_scope: bool = field(default=True)
    """
    True if the visitor is processing the "global" scope of the program, i.e.
    not inside block/functions.
    """

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not Const.check(node.annotation):
            if not self.global_scope and isinstance(node.target, ast.Name):
                if node.target.id in self.constants:
                    self.constants[node.target.id].shadowed = True
        else:
            if not self.global_scope:
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"All constant expressions have to be created in the global scope."
                )

            if not isinstance(node.target, ast.Name):
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"All constant expressions have to be assigned to 'ast.Name' nodes."
                )

            name = node.target.id
            if name in self.constants:
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"Constant '{name}' is already defined in the program.")
            try:
                # TODO: This does not take care of cases like:
                # ```
                # a: Const[i32] = 12
                # b: Const[i32] = a + 23
                # ```
                # It is not difficult to add support for this in th future
                # though.
                value = eval(ast.unparse(node.value))
                # For now, support primitive types only and add a guard to abort
                # in other cases.
                if isinstance(value, int) or isinstance(value, float):
                    self.constants[name] = Constant(value)
                else:
                    raise CodeGenerationException(
                        node.lineno, node.col_offset,
                        f"Constant '{name}' has evaluated type '{type(value)}' which is not supported."
                    )
            except Exception:
                # TODO: This error message can be improved by matching exact
                # exceptions returned by `eval` call.
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"Non-constant expression cannot be assigned to constant variable '{name}' or cannot be evaluated."
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Assignments are allowed to exactly one variable only.")
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            # Ensure that unless the variable is shadowed by local variables, it
            # cannot be assigned to.
            if name in self.constants and not self.constants[name].shadowed:
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"Cannot assign to constant variable '{name}'.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Record all variables that were shadowed before.
        previously_shadowed_constants: Set[str] = []
        for name , constant in self.constants.items():
            if constant.shadowed:
                previously_shadowed_constants.add(name)

        old_scope = self.global_scope
        self.global_scope = False
        for stmt in node.body:
            self.visit(stmt)
        self.global_scope = old_scope

        # Unshadow newly shadowed variables.
        for name in self.constants.keys():
            if name not in previously_shadowed_constants:
                self.constants[name].shadowed = False
