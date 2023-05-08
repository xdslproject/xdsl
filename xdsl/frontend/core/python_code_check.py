import ast
from dataclasses import dataclass, field
from typing import Dict, Set, Sequence
from xdsl.frontend.block import is_block
from xdsl.frontend.const import is_constant
from xdsl.frontend.exception import CodeGenerationException


@dataclass
class PythonCodeCheck:
    @staticmethod
    def run(stmts: Sequence[ast.stmt], file: str | None) -> None:
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
        CheckStructure.run_with_scope(single_scope, stmts, file)

        # Check constant/global variables are correctly defined. Should be
        # called only after the structure is checked.
        CheckAndInlineConstants.run(stmts, file)


@dataclass
class CheckStructure:
    @staticmethod
    def run_with_scope(
        single_scope: bool, stmts: Sequence[ast.stmt], file: str | None
    ) -> None:
        if single_scope:
            visitor = SingleScopeVisitor(file)
        else:
            visitor = MultipleScopeVisitor(file)
        for stmt in stmts:
            visitor.visit(stmt)


@dataclass
class SingleScopeVisitor(ast.NodeVisitor):
    file: str | None = field(default=None)
    """File path for error reporting."""

    block_names: Set[str] = field(default_factory=set)
    """Tracks duplicate block labels."""

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        assert is_block(node)

        if node.name in self.block_names:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Block '{node.name}' is already defined.",
            )
        self.block_names.add(node.name)

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if is_block(stmt):
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Cannot have a nested block '{stmt.name}' inside the "
                        f"block '{node.name}'.",
                    )
                else:
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Cannot have an inner function '{stmt.name}' inside "
                        f"the block '{node.name}'.",
                    )


@dataclass
class MultipleScopeVisitor(ast.NodeVisitor):
    file: str | None = field(default=None)

    function_and_block_names: Dict[str, Set[str]] = field(default_factory=dict)
    """Tracks duplicate function names and duplicate block labels."""

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        assert not is_block(node)

        if node.name in self.function_and_block_names:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Function '{node.name}' is already defined.",
            )
        self.function_and_block_names[node.name] = set()

        # Functions cannot have inner functions but can have blocks inside
        # which we still have to check.
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if not is_block(stmt):
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Cannot have an inner function '{stmt.name}' inside "
                        f"the function '{node.name}'.",
                    )
                else:
                    if stmt.name in self.function_and_block_names[node.name]:
                        raise CodeGenerationException(
                            self.file,
                            stmt.lineno,
                            stmt.col_offset,
                            f"Block '{stmt.name}' is already defined in "
                            f"function '{node.name}'.",
                        )
                    self.function_and_block_names[node.name].add(stmt.name)

                    for inner in stmt.body:
                        if isinstance(inner, ast.FunctionDef):
                            if is_block(inner):
                                raise CodeGenerationException(
                                    self.file,
                                    inner.lineno,
                                    inner.col_offset,
                                    f"Cannot have a nested block '{inner.name}'"
                                    f" inside the block '{stmt.name}'.",
                                )
                            else:
                                raise CodeGenerationException(
                                    self.file,
                                    inner.lineno,
                                    inner.col_offset,
                                    f"Cannot have an inner function '{inner.name}'"
                                    f" inside the block '{stmt.name}'.",
                                )


@dataclass
class CheckAndInlineConstants:
    """
    This class is responsible for checking that the constants defined in the
    frontend program are valid. Every valid constant is inlined as a new AST
    node.

    The algorithm for checking and inlining is iterative. When a new constant
    definition is encountered, the algorithm tries to inline it. This way
    frontend programs can define constants such as:

    ```
    a: Const[i32] = 1 + len([1, 2, 3, 4])
    b: Const[i32] = a * a
    # here b = 25
    ```

    Note that the algorithm does not remove constant definitions from the AST,
    but this functionality can be added later.
    """

    @staticmethod
    def run(stmts: Sequence[ast.stmt], file: str | None) -> None:
        CheckAndInlineConstants.run_with_variables(stmts, set(), file)

    @staticmethod
    def run_with_variables(
        stmts: Sequence[ast.stmt], defined_variables: Set[str], file: str | None
    ) -> None:
        for i, stmt in enumerate(stmts):
            # This variable (`a = ...`) can be redefined as a constant, and so
            # we have to keep track of these to raise an exception.
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                defined_variables.add(stmt.targets[0].id)
                continue

            # Similarly, this case (`a: i32 = ...`) can also be redefined as a
            # constant.
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and not is_constant(stmt.annotation)
            ):
                defined_variables.add(stmt.target.id)
                continue

            # This is a constant.
            if isinstance(stmt, ast.AnnAssign) and is_constant(stmt.annotation):
                if not isinstance(stmt.target, ast.Name):
                    raise CodeGenerationException(
                        file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"All constant expressions have to be assigned to "
                        "'ast.Name' nodes.",
                    )

                name = stmt.target.id
                try:
                    assert stmt.value is not None
                    value = eval(ast.unparse(stmt.value))
                except Exception:
                    # TODO: This error message can be improved by matching exact
                    # exceptions returned by `eval` call.
                    raise CodeGenerationException(
                        file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Non-constant expression cannot be assigned to "
                        f"constant variable '{name}' or cannot be evaluated.",
                    )

                # For now, support primitive types only and add a guard to abort
                # in other cases.
                if not isinstance(value, int) and not isinstance(value, float):
                    raise CodeGenerationException(
                        file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Constant '{name}' has evaluated type '{type(value)}' "
                        "which is not supported.",
                    )

                # TODO: We should typecheck the value against the type. This can
                # get tricky since ints can overflow, etc. For example, `a:
                # Const[i16] = 100000000` should give an error.
                new_node = ast.Constant(value)
                inliner = ConstantInliner(name, new_node, file)
                for candidate in stmts[(i + 1) :]:
                    inliner.visit(candidate)

                # Ideally, we can prune this AST node now, but it is easier just
                # to avoid it during code generation phase.
                continue

            # In case of a function/block definition, we must ensure we process
            # the nested list of statements as well. Note that if we reached
            # this then all constants above `i` must have been already inlined.
            # Hence, it is sufficient to check the function body only.
            if isinstance(stmt, ast.FunctionDef):
                new_defined_variables = set([arg.arg for arg in stmt.args.args])
                CheckAndInlineConstants.run_with_variables(
                    stmt.body, new_defined_variables, file
                )


@dataclass
class ConstantInliner(ast.NodeTransformer):
    """
    Given the name of a constant and a corresponding AST node, `ConstantInliner`
    traverses the AST and replaces the uses of the `name` with the node.
    Additionally, it is responsible for performing various checks whether the
    constant value is correctly used. In cases of a misuse (e.g. assigning to a
    constant), an exception is raised.
    """

    name: str
    """The name of the constant to inline."""

    new_node: ast.Constant
    """New AST node to inline."""

    file: str | None = field(default=None)
    """Path to the file containing the program."""

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == self.name
        ):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Constant '{self.name}' is already defined and cannot be "
                "assigned to.",
            )
        node.value = self.visit(node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if isinstance(node.target, ast.Name) and node.target.id == self.name:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Constant '{self.name}' is already defined.",
            )
        assert node.value is not None
        node.value = self.visit(node.value)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        for arg in node.args.args:
            if arg.arg == self.name:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    f"Constant '{self.name}' is already defined and cannot be "
                    "used as a function/block argument name.",
                )
        for stmt in node.body:
            self.visit(stmt)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name | ast.Constant:
        if node.id == self.name:
            return self.new_node
        else:
            return node
