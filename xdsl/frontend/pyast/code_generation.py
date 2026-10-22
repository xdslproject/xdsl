import ast
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import xdsl.dialects.builtin as builtin
import xdsl.dialects.cf as cf
import xdsl.dialects.func as func
import xdsl.dialects.py as py
import xdsl.dialects.scf as scf
import xdsl.dialects.symref as symref
from xdsl.frontend.pyast.utils.exceptions import (
    CodeGenerationException,
)
from xdsl.frontend.pyast.utils.op_inserter import OpInserter
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.utils.lexer import Location

FunctionDefinitionConstructor = Callable[
    [str, builtin.FunctionType, Region, Location], Operation
]
"""Wrap a function region containing argument bindings, before visiting statements."""

ReturnConstructor = Callable[[Sequence[SSAValue], Location], Operation | None]
"""Construct a return operation, or return None when no terminator is needed."""


def build_func(
    name: str, signature: builtin.FunctionType, body: Region, location: Location
) -> func.FuncOp:
    return func.FuncOp(name, signature, body)


def build_return(values: Sequence[SSAValue], location: Location) -> func.ReturnOp:
    return func.ReturnOp(*values)


@dataclass(init=False)
class CodeGenerationVisitor(ast.NodeVisitor):
    """
    Visitor that generates the `py` dialect from the Python AST, with source locations
    for the errors found when lowering it.
    """

    type_converter: TypeConverter
    """Used for type conversion during code generation."""

    inserter: OpInserter
    """Used for inserting newly generated operations to the right block."""

    symbol_table: set[str] | None = field(default=None)
    """
    The names of the local variables. A single set is sufficient because inner
    functions and global variables are not allowed (yet).
    """

    last_return: ast.Return | None = field(default=None)
    """The most recently visited return in the function being generated."""

    function_body: Block | None = field(default=None)
    """
    The entry block of the function being generated. Statements in nested regions,
    such as the branches of an `if`, are inserted into other blocks.
    """

    file: str | None
    """Path of the file containing the program being processed."""

    function_definition_constructor: FunctionDefinitionConstructor
    """Construct an enclosing operation before visiting the function statements."""

    return_constructor: ReturnConstructor
    """Construct return operations, optionally omitting the terminator."""

    def __init__(
        self,
        type_converter: TypeConverter,
        module: builtin.ModuleOp,
        file: str | None,
        function_definition_constructor: FunctionDefinitionConstructor = build_func,
        return_constructor: ReturnConstructor = build_return,
    ) -> None:
        self.type_converter = type_converter
        self.file = file
        self.function_definition_constructor = function_definition_constructor
        self.return_constructor = return_constructor

        assert len(module.body.blocks) == 1
        self.inserter = OpInserter(module.body.block)

    def location(self, node: ast.expr | ast.stmt | ast.arg) -> builtin.FileLineColLoc:
        """The location of `node` in the source, with a one-based column."""
        return builtin.FileLineColLoc(
            builtin.StringAttr(self.file or "<unknown>"),
            builtin.IntAttr(node.lineno),
            builtin.IntAttr(node.col_offset + 1),
        )

    def insert_op(self, op: Operation, node: ast.expr | ast.stmt) -> None:
        """Insert an operation generated from `node`."""
        op.location = self.location(node)
        self.inserter.insert_op(op)

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_single_value(self, node: ast.expr) -> SSAValue:
        """Visit an expression that must produce exactly one SSA value."""
        stack_size = len(self.inserter.stack)
        self.visit(node)
        if len(self.inserter.stack) != stack_size + 1:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Expected an expression with exactly one result.",
            )
        return self.inserter.get_operand()

    def visit_condition(self, node: ast.expr) -> SSAValue:
        """Visit an expression used as a condition, giving its truth value."""
        self.insert_op(py.PyBoolOp(self.visit_single_value(node)), node)
        return self.inserter.get_operand()

    def generic_visit(self, node: ast.AST) -> None:
        raise CodeGenerationException(
            self.file,
            getattr(node, "lineno"),
            getattr(node, "col_offset"),
            f"Unsupported Python AST node {node!s}",
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # TODO: Implement assignemnt in the next patch.
        pass

    def visit_Assert(self, node: ast.Assert) -> None:
        condition = self.visit_condition(node.test)
        if node.msg is None:
            msg = ""
        else:
            if not isinstance(node.msg, ast.Constant) or not isinstance(
                node.msg.value, str
            ):
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Expected a string constant for assertion message, found "
                    f"'ast.{type(node.msg).__qualname__}'",
                )
            msg = str(node.msg.value)
        self.insert_op(cf.AssertOp(condition, msg), node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Only assignment to a single variable is supported.",
            )
        # Desymrefication cannot yet carry an updated value out of a nested region,
        # so an assignment inside an `if` would silently keep the old value.
        if self.inserter.insertion_point is not self.function_body:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Assignments are only supported directly in the function body.",
            )
        value = self.visit_single_value(node.value)
        name = node.targets[0].id
        assert self.symbol_table is not None
        if name not in self.symbol_table:
            self.symbol_table.add(name)
            self.insert_op(symref.DeclareOp(name), node)
        self.insert_op(symref.UpdateOp(name, value), node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.visit(node.right)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()
        op_name = builtin.StringAttr(node.op.__class__.__qualname__.lower())
        self.insert_op(py.PyBinOp(op_name, lhs, rhs), node)

    def visit_Call(self, node: ast.Call) -> None:
        match node.func:
            case ast.Name():
                callee = node.func.id
            case ast.Attribute(value=ast.Name(id=class_name), attr=method_name):
                callee = f"{class_name}.{method_name}"
            case _:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Unsupported call expression.",
                )

        args = [self.visit_single_value(arg) for arg in node.args]
        keywords: list[str] = []
        for keyword in node.keywords:
            if keyword.arg is None:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Unpacking keyword arguments is not supported.",
                )
            keywords.append(keyword.arg)
            args.append(self.visit_single_value(keyword.value))

        self.insert_op(py.PyCallOp(callee, args, keywords), node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Allow a single comparison only.
        if len(node.comparators) != 1 or len(node.ops) != 1:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Expected a single comparator, but found {len(node.comparators)}.",
            )
        comp = node.comparators[0]
        op_name: str = node.ops[0].__class__.__qualname__

        # Table with currently unsupported Python AST cmpops.
        # The "is" and "is not" operators are (currently) not supported,
        # since the frontend does not consider/preserve object identity.
        # Finally, "not in" does not directly correspond to a special method
        # and is instead simply implemented as the negation of __contains__
        # which the current mapping framework cannot handle.
        unsupported_python_AST_cmpop = {"Is", "IsNot", "NotIn"}

        if op_name in unsupported_python_AST_cmpop:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Unsupported comparison operation '{op_name}'.",
            )

        self.visit(comp)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()
        self.insert_op(
            py.PyCompareOp(builtin.StringAttr(op_name.lower()), lhs, rhs), node
        )

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, int | float | str):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Unsupported constant '{node.value}' of type '{type(node.value).__qualname__}'.",
            )
        self.insert_op(py.PyConstOp.from_value(node.value), node)

    def visit_Expr(self, node: ast.Expr) -> None:
        stack_size = len(self.inserter.stack)
        self.visit(node.value)
        # Keep the operations, but discard the values of an expression statement.
        del self.inserter.stack[stack_size:]

    def visit_For(self, node: ast.For) -> None:
        raise NotImplementedError("For loops are currently not supported!")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Set the symbol table.
        if self.symbol_table is not None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Cannot have an inner function '{node.name}' inside another function.",
            )
        self.symbol_table = set()
        self.last_return = None

        # The annotations are kept as written, and resolved when lowering.
        argument_annotations: list[str] = []
        for arg in node.args.args:
            if arg.annotation is None:
                raise CodeGenerationException(
                    self.file,
                    arg.lineno,
                    arg.col_offset,
                    "Function arguments must be type hinted",
                )
            argument_annotations.append(ast.unparse(arg.annotation))

        returns = node.returns
        if returns is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Function return type must be annotated.",
            )

        entry_block = Block(arg_types=[py.PyObjectType()] * len(node.args.args))
        self.function_body = entry_block
        parent_block = self.inserter.insertion_point
        self.inserter.set_insertion_point_from_block(entry_block)

        # All arguments are declared using symref.
        for arg, value in zip(node.args.args, entry_block.args, strict=True):
            symbol_name = str(arg.arg)
            value.name_hint = symbol_name
            value.location = self.location(arg)
            self.symbol_table.add(symbol_name)
            entry_block.add_op(symref.DeclareOp(symbol_name))
            entry_block.add_op(symref.UpdateOp(symbol_name, value))

        function_op = py.PyFuncOp(
            node.name, argument_annotations, ast.unparse(returns), Region(entry_block)
        )
        function_op.location = self.location(node)
        parent_block.add_op(function_op)

        # Parse function body.
        has_docstring = ast.get_docstring(node, clean=False) is not None
        visit_start_index = 1 if has_docstring else 0
        for stmt in itertools.islice(node.body, visit_start_index, None):
            self.visit(stmt)

        if self.last_return is not None and self.last_return is not node.body[-1]:
            raise CodeGenerationException(
                self.file,
                self.last_return.lineno,
                self.last_return.col_offset,
                "Return statement must be the final statement directly "
                "in the function body.",
            )

        if self.last_return is None:
            self.visit(ast.copy_location(ast.Return(value=None), node))

        # When function definition is processed, reset the symbol table and set
        # the insertion point.
        self.symbol_table = None
        self.function_body = None
        self.inserter.set_insertion_point_from_block(parent_block)

    def visit_If(self, node: ast.If) -> None:
        # Get the condition.
        cond = self.visit_condition(node.test)
        cond_block = self.inserter.insertion_point

        def visit_region(stmts: list[ast.stmt]) -> Region:
            region = Region([Block()])
            self.inserter.set_insertion_point_from_region(region)
            for stmt in stmts:
                self.visit(stmt)
            return region

        # Generate code for both branches.
        true_region = visit_region(node.body)
        false_region = visit_region(node.orelse)

        # In our case, if statement never returns a value and therefore we can
        # simply yield nothing. It is the responsibility of subsequent passes to
        # ensure SSA-form of IR and that values are yielded correctly.
        true_region.blocks[-1].add_op(scf.YieldOp())
        false_region.blocks[-1].add_op(scf.YieldOp())
        op = scf.IfOp(cond, [], true_region, false_region)

        # Reset insertion point and insert a new operation.
        self.inserter.set_insertion_point_from_block(cond_block)
        self.insert_op(op, node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        cond = self.visit_condition(node.test)
        cond_block = self.inserter.insertion_point

        def visit_expr(expr: ast.expr) -> Region:
            region = Region([Block()])
            self.inserter.set_insertion_point_from_region(region)
            self.visit(expr)
            result = self.inserter.get_operand()
            self.insert_op(scf.YieldOp(result), expr)
            return region

        # Generate code for both branches. Their types are checked when lowering.
        true_region = visit_expr(node.body)
        false_region = visit_expr(node.orelse)
        op = scf.IfOp(cond, [py.PyObjectType()], true_region, false_region)

        # Reset insertion point to add scf.if.
        self.inserter.set_insertion_point_from_block(cond_block)
        self.insert_op(op, node)

    def visit_Name(self, node: ast.Name) -> None:
        assert self.symbol_table is not None
        if node.id not in self.symbol_table:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Symbol '{node.id}' is not defined.",
            )
        self.insert_op(symref.FetchOp(node.id, py.PyObjectType()), node)

    def visit_Pass(self, node: ast.Pass) -> None:
        pass

    def visit_Return(self, node: ast.Return) -> None:
        if self.last_return is not None:
            raise CodeGenerationException(
                self.file,
                self.last_return.lineno,
                self.last_return.col_offset,
                "Return statement must be the final statement directly "
                "in the function body.",
            )
        self.last_return = node

        value = node.value
        if value is None or (isinstance(value, ast.Constant) and value.value is None):
            operands = ()
        else:
            # TODO: Support multiple return values if we allow multiple assignments.
            operands = (self.visit_single_value(value),)
        self.insert_op(py.PyReturnOp(*operands), node)
