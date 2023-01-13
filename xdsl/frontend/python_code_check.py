import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from xdsl.frontend.block import is_block
from xdsl.frontend.const import Const
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.type_conversion import TypeConverter
from xdsl.ir import Attribute


@dataclass
class PythonCodeCheck:

    @staticmethod
    def run_with_type_converter(type_converter: TypeConverter, stmts: List[ast.stmt]) -> None:
        """
        Checks if Python code within `CodeContext` is supported. On unsupported
        cases, an exception is raised. Particularly, the check is used for
        scoping because Python (as an interpreted language) does not have a
        notion of constants, global variables, etc. Hence, one can redefine
        functions, place variables in arbitrary locations and do many other
        weird things which xDSL/MLIR would not like.
        """

        # Any code written within code context can be organized either a
        # sequence of instructions (interpreted one by one), for example:
        # ```
        # with CodeContext(p):
        #   a: i32 = 45
        #   b: i32 = 23
        #   result: i32 = a + b
        # ```
        # Alternatively, the code can be viewed as sequence of functions
        # (possibly with a dedicated entry point), for example:
        # ```
        # with CodeContext(p):
        #  def foo(x: i32) -> i32:
        #    return x
        #  def main():
        #    y: i32 = foo(100)
        # ```
        # First, find out which of the two modes is used.
        interpreter_mode: bool = True
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef) and not is_block(stmt):
                interpreter_mode = False
                break

        # Check Python code is valid for compilation/execution based on the
        # current code mode.
        visitor = SingleScopeVisitor(type_converter) if interpreter_mode else MultipleScopeVisitor(type_converter)
        for stmt in stmts:
            visitor.visit(stmt)


@dataclass
class SingleScopeVisitor(ast.NodeVisitor):

    constants: Dict[str, (Attribute, Any)] = field(default_factory=dict)
    """Stores constants of the current program and their values."""

    block_names: Set[str] = field(default_factory=set)
    """Tracks duplicate block labels."""

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if Const.check(node.annotation):
            assert isinstance(node.target, ast.Name)
            try:
                self.constants[node.target.id] = eval(ast.unparse(node.value))
            except Exception:
                raise CodeGenerationException(node.lineno, node.col_offset, f"Non-constant expression cannot be assigned to constant variable '{node.target.id}'.")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Assignments are allowed to exaclty one variable only.")
        if isinstance(node.targets[0], ast.Name) and (name := node.targets[0].id) in self.constants:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Cannot assign to constant variable '{name}'.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        assert is_block(node)

        if node.name in self.block_names:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Block '{node.name}' is already defined.")
        self.block_names.add(node.name)

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if is_block(stmt):
                    raise CodeGenerationException(stmt.lineno, stmt.col_offset, f"Cannot have a nested block '{stmt.name}' inside the block '{node.name}'.")
                else:
                    raise CodeGenerationException(stmt.lineno, stmt.col_offset, f"Cannot have an inner function '{stmt.name}' inside the block '{node.name}'.")


@dataclass
class MultipleScopeVisitor(ast.NodeVisitor):

    constants: Dict[str, (Attribute, Any)] = field(default_factory=dict)
    """Stores constants of the current program and their values."""

    function_and_block_names: Dict[str, Set[str]] = field(default_factory=dict)
    """Tracks duplicate function names and duplicate block labels."""

    visiting_function_def: bool = field(default=False)
    """If set, the visitor is currently visiting the function definition."""

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Const assignement inside the function.
        # TODO: We can actually support function-level compile-time constants,
        # but it was never a priority to implement.
        if self.visiting_function_def and Const.check(node.annotation):
                raise CodeGenerationException(node.lineno, node.col_offset, f"Cannot create constants inside functions. Try to remove the '{Const.__name__}' annotation or use a constant defined outside of the function.")

        # Non-const assignement outside the function.
        if not self.visiting_function_def and not Const.check(node.annotation):
            raise CodeGenerationException(node.lineno, node.col_offset, f"Cannot create the variable with non-constant type. Try to wrap it using '{Const.__name__}'.")

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.visiting_function_def:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Cannot assign to the variable outside of the function definition. If you want to use the variable as a compile-time constant, wrap the type into '{Const.__name__}'.")
        
        # Otherwise, we are checking the function body.
        if len(node.targets) != 1:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Assignments are allowed to exaclty one variable only.")
        if isinstance(node.targets[0], ast.Name) and (name := node.targets[0].id) in self.constants:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Cannot assign to constant variable '{name}'.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        assert not is_block(node)

        if node.name in self.function_and_block_names:
            raise CodeGenerationException(node.lineno, node.col_offset, f"Function '{node.name}' is already defined.")
        self.function_and_block_names[node.name] = set()

        # Functions cannot have inner functions but can have blocks inside
        # which we still have to check.
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if not is_block(stmt):
                    raise CodeGenerationException(stmt.lineno, stmt.col_offset, f"Cannot have an inner function '{stmt.name}' inside the function '{node.name}'.")
                else:
                    if stmt.name in self.function_and_block_names[node.name]:
                        raise CodeGenerationException(stmt.lineno, stmt.col_offset, f"Block '{stmt.name}' is already defined in function '{node.name}'.")
                    self.function_and_block_names[node.name].add(stmt.name)

                    for inner_stmt in stmt.body:
                        if isinstance(inner_stmt, ast.FunctionDef):
                            if is_block(inner_stmt):
                                raise CodeGenerationException(inner_stmt.lineno, inner_stmt.col_offset, f"Cannot have a nested block '{inner_stmt.name}' inside the block '{stmt.name}'.")
                            else:
                                raise CodeGenerationException(inner_stmt.lineno, inner_stmt.col_offset, f"Cannot have an inner function '{inner_stmt.name}' inside the block '{stmt.name}'.")
