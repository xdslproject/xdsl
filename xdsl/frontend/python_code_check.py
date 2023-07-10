import ast
from dataclasses import dataclass, field
from typing import Sequence

from xdsl.frontend.block import is_block
from xdsl.frontend.const import is_constant, is_constant_stmt
from xdsl.frontend.exception import CodeGenerationException

# Type aliases for simplicity.
BlockMap = dict[str, ast.FunctionDef]
FunctionData = tuple[ast.FunctionDef, BlockMap]
FunctionMap = dict[str, FunctionData]


@dataclass
class PythonCodeCheck:
    @staticmethod
    def run(stmts: Sequence[ast.stmt], file: str | None) -> FunctionMap:
        """
        Checks if Python code within `CodeContext` is supported. On unsupported
        cases, an exception is raised.

        Performed checks and transformations:

            1. Checks structure of code inside `CodeContext`. For example, no
               inner functions are allowed, etc. For more information see the
               docstring of `CheckStructure`.

            2. Checks the placement of constant expressions and inlines them
               into the AST.
        """
        # Check Python code is correctly structured.
        checker = CheckStructure(file)
        checker.run(stmts)

        # Check constant expressions are correctly defined. Should be called
        # only after the structure is checked.
        CheckAndInlineConstants.run(stmts, file)

        # Return well-structured functions and blocks to the caller.
        return checker.functions_and_blocks


@dataclass
class CheckStructure:
    """
    Ensures that the front-end program can be lowered to xDSL.

    Any code written within `CodeContext` must be organized as a sequence of
    functions (possibly with a dedicated entry point), for example:

    ```
    with CodeContext(p):
        def foo(x: i32) -> i32:
            return x
        def bar():
            y: i32 = foo(100)
            return

        def main():
            bar()
            return
    ```

    For each function, it holds that:
        1) Any function does not contain inner functions.
        2) Any function has an explicit terminator: a `return` statement.

    Additionally, any function can contain explicitly defined blocks, for
    example:

    ```
    with CodeContext(p):
        def foo(x: i32) -> i32:
            @block
            def bb0(y: i32) -> i32:
                # unconditional branch to another block
                return bb1(y)

            @block
            def bb1(y: i32) -> i32:
                # terminator for the function
                return y

        # specifies the entry block
        return bb0(x)
    ```

    For each block, it holds that:
        1) No block has inner functions or nested blocks.
        2) Any block has an explicit terminator: a `return` statement. The
           terminator can either transfer control flow to a next block, or
           terminate the enclosing function.
        3) It is up to a user to ensure that the control flow is transfered
           correctly, e.g. to avoid infinite cycles.
    """

    file: str | None = field(default=None)
    """File for error reporting."""

    functions_and_blocks: FunctionMap = field(default_factory=dict)
    """
    Contains all information about functions and blocks. Populated during the
    structure check.
    """

    def run(self, stmts: Sequence[ast.stmt]) -> None:
        for stmt in stmts:
            # Allow constant expression statements or pass.
            if is_constant_stmt(stmt) or isinstance(stmt, ast.Pass):
                continue

            # TODO: Right now we want all code to be placed in functions to make
            # code generation easier. This is limiting but can be fixed easily by
            # placing the whole AST into a dummy function, and performing code
            # generation on it. The only challenge is to make error messages
            # consistent.
            if isinstance(stmt, ast.FunctionDef):
                # Function should be a top-level operation.
                if is_block(stmt):
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Expected a function, but found a block '{stmt.name}'."
                        " Only functions can be declared in the CodeContext",
                    )

                # Record this function.
                if stmt.name in self.functions_and_blocks:
                    line = self.functions_and_blocks[stmt.name][0].lineno
                    col = self.functions_and_blocks[stmt.name][0].col_offset
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Function '{stmt.name}' is already defined at line "
                        f"{line} column {col}.",
                    )
                self.functions_and_blocks[stmt.name] = (stmt, dict())

                # Every function must have an explicit terminator, i.e. a return
                # statement. Using pass is not allowed. This design makes code
                # generation easier and can be relaxed in the future.
                if len(stmt.body) == 0:
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Function '{stmt.name}' must have an explicit terminator."
                        " Have you tried adding a return statement?",
                    )
                if not isinstance(stmt.body[-1], ast.Return):
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Function '{stmt.name}' must have an explicit return"
                        " in the end.",
                    )

                # Lastly, record basic block information that we can check
                # afterwards.
                for inner_stmt in stmt.body:
                    if isinstance(inner_stmt, ast.FunctionDef) and is_block(inner_stmt):
                        if inner_stmt.name in self.functions_and_blocks[stmt.name][1]:
                            line = self.functions_and_blocks[stmt.name][1][
                                inner_stmt.name
                            ].lineno
                            col = self.functions_and_blocks[stmt.name][1][
                                inner_stmt.name
                            ].col_offset
                            raise CodeGenerationException(
                                self.file,
                                stmt.lineno,
                                stmt.col_offset,
                                f"Block '{inner_stmt.name}' is already defined at line "
                                f"{line} column {col}.",
                            )
                        self.functions_and_blocks[stmt.name][1][
                            inner_stmt.name
                        ] = inner_stmt
                continue

            # Otherwise, not a function, pass nor constant expression. Abort.
            raise CodeGenerationException(
                self.file,
                stmt.lineno,
                stmt.col_offset,
                "Frontend program must consist of functions or constant "
                "expressions.",
            )

        # Check structure of all functions and if necessary populate the map
        # with block information.
        for function_data in self.functions_and_blocks.values():
            self._check_function_structure(function_data[0])

    def _is_branch(self, function_name: str, node: ast.expr | None) -> bool:
        """Returns true if the terminator node is an unconditional branch."""
        return (
            node is not None
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.functions_and_blocks[function_name][1]
        )

    def _is_cond_branch(self, function_name: str, node: ast.expr | None) -> bool:
        """Returns true if the terminator node is a conditional branch."""
        return (
            node is not None
            and isinstance(node, ast.IfExp)
            and isinstance(node.body, ast.Call)
            and isinstance(node.body.func, ast.Name)
            and node.body.func.id in self.functions_and_blocks[function_name][1]
            and isinstance(node.orelse, ast.Call)
            and isinstance(node.orelse.func, ast.Name)
            and node.orelse.func.id in self.functions_and_blocks[function_name][1]
        )

    def _check_block_structure(self, function_name: str, node: ast.FunctionDef) -> bool:
        # Check that the basic block is well-formed.
        for stmt in node.body:
            # No inner functions or nested blocks.
            if isinstance(stmt, ast.FunctionDef):
                if is_block(stmt):
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Cannot have a nested block '{stmt.name}'"
                        f" inside the block '{node.name}'.",
                    )
                else:
                    raise CodeGenerationException(
                        self.file,
                        stmt.lineno,
                        stmt.col_offset,
                        f"Cannot have a nested function '{stmt.name}'"
                        f" inside the block '{node.name}'.",
                    )

        # Check blocks have an explicit terminator.
        if len(node.body) == 0 or not isinstance(node.body[-1], ast.Return):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Block '{node.name}' must have an explicit terminator."
                " Have you tried adding a return statement?",
            )

        # Check if the terminator of the block is well-formed. It can
        # terminate function, branch unconditionally to another block, or be
        # a conditional branch.
        assert isinstance(node.body[-1], ast.Return)
        terminator = node.body[-1].value
        return self._is_branch(function_name, terminator) or self._is_cond_branch(
            function_name, terminator
        )

    def _check_function_structure(self, node: ast.FunctionDef):
        # Functions cannot have inner functions but can have blocks inside
        # which we still have to check.
        num_explicit_blocks = len(self.functions_and_blocks[node.name][1])
        num_explicit_blocks_with_branches = 0

        for stmt in node.body[:-1]:
            # Constant expressions can be placed inside the function.
            if is_constant_stmt(stmt):
                continue

            # If there are explicit blocks, no operations are allowed outside of
            # them.
            if num_explicit_blocks > 0 and not isinstance(stmt, ast.FunctionDef):
                raise CodeGenerationException(
                    self.file,
                    stmt.lineno,
                    stmt.col_offset,
                    f"Function '{node.name}' cannot contain operations outside"
                    " of blocks apart from explicit entry point or constant "
                    "expressions.",
                )

            # Otherwise we allow anything, and only have to carefully look at
            # inner functions.
            if not isinstance(stmt, ast.FunctionDef):
                continue

            # Only blocks are allowed
            if not is_block(stmt):
                raise CodeGenerationException(
                    self.file,
                    stmt.lineno,
                    stmt.col_offset,
                    f"Cannot have an inner function '{stmt.name}' inside "
                    f"the function '{node.name}'.",
                )

            # Check the block and record if its terminator is a branch or not.
            if self._check_block_structure(node.name, stmt):
                num_explicit_blocks_with_branches += 1

        # Last check: we must have exactly one terminating block if blocks are
        # explicitly defined.
        if (
            num_explicit_blocks > 1
            and num_explicit_blocks == num_explicit_blocks_with_branches
        ):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Function '{node.name}' does not have a terminating block.",
            )
        num_explicit_terminating_blocks = (
            num_explicit_blocks - num_explicit_blocks_with_branches
        )
        if num_explicit_terminating_blocks > 1:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Function '{node.name}' expected one terminating block, got"
                f" {num_explicit_terminating_blocks}.",
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
    """

    @staticmethod
    def run(stmts: Sequence[ast.stmt], file: str | None) -> None:
        CheckAndInlineConstants.run_with_variables(stmts, set(), file)

    @staticmethod
    def run_with_variables(
        stmts: Sequence[ast.stmt], defined_variables: set[str], file: str | None
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
                continue

            # In case of a function/block definition, we must ensure we process
            # the nested list of statements as well. Note that if we reached
            # this then all constants above `i` must have been already inlined.
            # Hence, it is sufficient to check the function body only.
            if isinstance(stmt, ast.FunctionDef):
                new_defined_variables = {arg.arg for arg in stmt.args.args}
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
