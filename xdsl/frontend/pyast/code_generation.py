import ast
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import xdsl.dialects.builtin as builtin
import xdsl.dialects.cf as cf
import xdsl.dialects.py as py
import xdsl.dialects.symref as symref
from xdsl.frontend.pyast.utils.exceptions import (
    CodeGenerationException,
)
from xdsl.frontend.pyast.utils.op_inserter import OpInserter
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue, TypeAttribute


@dataclass(init=False)
class CodeGenerationVisitor(ast.NodeVisitor):
    """
    Visitor that generates xDSL from the Python AST. Functions and returns are
    generated as `py.func` and `py.return`, for the registered constructors to be
    applied when lowering.
    """

    type_converter: TypeConverter
    """Used for type conversion during code generation."""

    inserter: OpInserter
    """Used for inserting newly generated operations to the right block."""

    symbol_table: dict[str, Attribute] | None = field(default=None)
    """
    Maps local variable names to their xDSL types. A single dictionary is sufficient
    because inner functions and global variables are not allowed (yet).
    """

    last_return: ast.Return | None = field(default=None)
    """The most recently visited return in the function being generated."""

    file: str | None
    """Path of the file containing the program being processed."""

    def __init__(
        self,
        type_converter: TypeConverter,
        module: builtin.ModuleOp,
        file: str | None,
    ) -> None:
        self.type_converter = type_converter
        self.file = file

        assert len(module.body.blocks) == 1
        self.inserter = OpInserter(module.body.block)

    def location(self, node: ast.expr | ast.stmt | ast.arg) -> builtin.FileLineColLoc:
        """The location of `node` in the source, with a one-based column."""
        return builtin.FileLineColLoc(
            builtin.StringAttr(self.file or "<unknown>"),
            builtin.IntAttr(node.lineno),
            builtin.IntAttr(node.col_offset + 1),
        )

    def insert_op(self, op: Operation, node: ast.expr | ast.stmt | ast.arg) -> None:
        """Insert an operation generated from `node`."""
        op.location = self.location(node)
        self.inserter.insert_op(op)

    def get_symbol(self, node: ast.Name) -> Attribute:
        assert self.symbol_table is not None
        if node.id not in self.symbol_table:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Symbol '{node.id}' is not defined.",
            )
        return self.symbol_table[node.id]

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

    def generic_visit(self, node: ast.AST) -> None:
        raise CodeGenerationException(
            self.file,
            getattr(node, "lineno"),
            getattr(node, "col_offset"),
            f"Unsupported Python AST node {node!s}",
        )

    def visit_Assert(self, node: ast.Assert) -> None:
        self.visit(node.test)
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
        self.insert_op(cf.AssertOp(self.inserter.get_operand(), msg), node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Only assignment to a single variable is supported.",
            )
        value = self.visit_single_value(node.value)
        name = node.targets[0].id
        assert self.symbol_table is not None
        if name not in self.symbol_table:
            self.symbol_table[name] = value.type
            self.insert_op(symref.DeclareOp(name), node)
        elif self.symbol_table[name] != value.type:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Cannot change the type of variable '{name}'.",
            )
        self.insert_op(symref.UpdateOp(name, value), node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        op_name: str = node.op.__class__.__qualname__

        # Table with mappings of Python AST operator to Python methods.
        python_AST_operator_to_python_overload = {
            "Add": "__add__",
            "Sub": "__sub__",
            "Mult": "__mul__",
            "Div": "__truediv__",
            "FloorDiv": "__floordiv__",
            "Mod": "__mod__",
            "Pow": "__pow__",
            "LShift": "__lshift__",
            "RShift": "__rshift__",
            "BitOr": "__or__",
            "BitXor": "__xor__",
            "BitAnd": "__and__",
            "MatMult": "__matmul__",
        }

        if op_name not in python_AST_operator_to_python_overload:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Unexpected binary operation {op_name}.",
            )

        # Check that the types of the operands are the same.
        # This is a (temporary?) restriction over Python for implementation simplicity.
        # This also means that we do not need to support reflected operations
        # (__radd__, __rsub__, etc.) which only exist for operations between different types.
        self.visit(node.right)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()
        if lhs.type != rhs.type:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Expected the same types for binary operation '{op_name}', "
                f"but got {lhs.type} and {rhs.type}.",
            )

        ir_type = cast(TypeAttribute, lhs.type)
        source_type = self.type_converter.type_registry.get_annotation(ir_type)
        if source_type is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"IR type '{ir_type}' is not registered with a source type.",
            )

        method_name = python_AST_operator_to_python_overload[op_name]
        function_name = f"{source_type.__qualname__}.{method_name}"
        op = self.type_converter.function_registry.resolve_operation(
            module_name=source_type.__module__,
            method_name=function_name,
            args=(lhs, rhs),
        )
        if op is not None:
            self.insert_op(op, node)
            return

        overload_name = python_AST_operator_to_python_overload[op_name]
        raise CodeGenerationException(
            self.file,
            node.lineno,
            node.col_offset,
            f"Binary operation '{op_name}' "
            f"is not supported by type '{source_type.__qualname__}' "
            f"which does not overload '{overload_name}'.",
        )

    def visit_Call(self, node: ast.Call) -> None:
        match node.func:
            case ast.Name():
                source_kind = "function"
                source, source_name = self._call_source_function(node)
            case ast.Attribute():
                source_kind = "classmethod"
                source, source_name = self._call_source_classmethod(node)
            case _:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Unsupported call expression.",
                )

        ir_op = self.type_converter.function_registry.get_operation_constructor(source)
        if ir_op is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"{source_kind.capitalize()} '{source_name}' is not registered.",
            )

        args = [self.visit_single_value(arg) for arg in node.args]
        kwargs: dict[str, SSAValue] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Unpacking keyword arguments is not supported.",
                )
            kwargs[keyword.arg] = self.visit_single_value(keyword.value)

        self.insert_op(ir_op(*args, **kwargs), node)

    # Get called function for a call expression.
    def _call_source_function(self, node: ast.Call) -> tuple[Callable[..., Any], str]:
        assert isinstance(node.func, ast.Name)

        func_name = node.func.id
        func = self.type_converter.globals.get(func_name, None)
        if func is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Function '{func_name}' is not defined in scope.",
            )
        return func, func_name

    # Get called classmethod for a call expression.
    def _call_source_classmethod(
        self, node: ast.Call
    ) -> tuple[Callable[..., Any], str]:
        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Name)

        class_name = node.func.value.id
        method_name = node.func.attr
        classmethod_name = f"{class_name}.{method_name}"

        source_class = self.type_converter.globals.get(class_name, None)
        if source_class is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Class '{class_name}' is not defined in scope.",
            )
        classmethod_ = getattr(source_class, method_name, None)
        if classmethod_ is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Method '{method_name}' is not defined on class '{class_name}'.",
            )
        return classmethod_, classmethod_name

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

        # Table with mappings of Python AST cmpop to Python method.
        python_AST_cmpop_to_python_overload = {
            "Eq": "__eq__",
            "Gt": "__gt__",
            "GtE": "__ge__",
            "Lt": "__lt__",
            "LtE": "__le__",
            "NotEq": "__ne__",
            "In": "__contains__",
            "NotIn": "__contains__",
        }

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

        # Check that the types of the operands are the same.
        # This is a (temporary?) restriction over Python for implementation simplicity.
        # This also means that we do not need to consider swapping arguments
        # (__eq__ and __ne__ are their own reflection, __lt__ <-> __gt__  and __le__ <-> __ge__).
        self.visit(comp)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()
        if lhs.type != rhs.type:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Expected the same types for comparison operator '{op_name}',"
                f" but got {lhs.type} and {rhs.type}.",
            )

        ir_type = cast(TypeAttribute, lhs.type)
        source_type = self.type_converter.type_registry.get_annotation(ir_type)
        if source_type is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"IR type '{ir_type}' is not registered with a source type.",
            )

        method_name = python_AST_cmpop_to_python_overload[op_name]
        function_name = f"{source_type.__qualname__}.{method_name}"
        op = self.type_converter.function_registry.resolve_operation(
            module_name=source_type.__module__,
            method_name=function_name,
            args=(lhs, rhs),
        )
        if op is not None:
            self.insert_op(op, node)
            return

        python_op = python_AST_cmpop_to_python_overload[op_name]
        raise CodeGenerationException(
            self.file,
            node.lineno,
            node.col_offset,
            f"Comparison operation '{op_name}' "
            f"is not supported by type '{ir_type.name}' "
            f"which does not overload '{python_op}'.",
        )

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            literal_op := self.type_converter.literal_registry.resolve_operation(
                node.value
            )
        ) is not None:
            self.insert_op(literal_op, node)
            return

        raise CodeGenerationException(
            self.file,
            node.lineno,
            node.col_offset,
            f"Unsupported constant '{node.value}' of type '{type(node.value).__qualname__}'.",
        )

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
        self.symbol_table = dict()
        self.last_return = None

        # Then, keep the annotations in the function signature, to be resolved when
        # lowering. The argument types are also needed now, for their variables.
        argument_types: list[Attribute] = []
        annotated_types: list[Attribute] = []
        for arg in node.args.args:
            if arg.annotation is None:
                raise CodeGenerationException(
                    self.file,
                    arg.lineno,
                    arg.col_offset,
                    "Function arguments must be type hinted",
                )
            annotation = ast.unparse(arg.annotation)
            xdsl_type = self.type_converter.type_registry.resolve_attribute(
                annotation, self.type_converter.globals
            )
            if xdsl_type is None:
                raise CodeGenerationException(
                    self.file,
                    arg.lineno,
                    arg.col_offset,
                    f"Unsupported function argument type: '{annotation}'",
                )
            argument_types.append(xdsl_type)
            annotated_types.append(py.PyObjectType(annotation))

        returns = node.returns
        if returns is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Function return type must be annotated.",
            )
        function_op = py.PyFuncOp(
            node.name,
            builtin.FunctionType.from_lists(
                annotated_types, [py.PyObjectType(ast.unparse(returns))]
            ),
            Region(Block(arg_types=annotated_types)),
        )
        self.insert_op(function_op, node)
        entry_block = function_op.body.block
        parent_block = self.inserter.insertion_point
        self.inserter.set_insertion_point_from_block(entry_block)

        # All arguments are declared using symref.
        for arg, value, xdsl_type in zip(
            node.args.args, entry_block.args, argument_types, strict=True
        ):
            symbol_name = str(arg.arg)
            value.name_hint = symbol_name
            self.symbol_table[symbol_name] = xdsl_type
            self.insert_op(symref.DeclareOp(symbol_name), arg)
            # In the body, arguments are plain objects whatever their annotation.
            plain = py.PyCastOp(value, py.PyObjectType())
            self.insert_op(plain, arg)
            self.insert_op(symref.UpdateOp(symbol_name, plain.res), arg)

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
        self.inserter.set_insertion_point_from_block(parent_block)

    def visit_Name(self, node: ast.Name) -> None:
        self.insert_op(symref.FetchOp(node.id, self.get_symbol(node)), node)

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
            # A bare return returns None, which is constructed first, as in bytecode.
            none = py.PyConstOp.from_value(None)
            self.insert_op(none, node)
            result = none.res
        else:
            # TODO: Support multiple return values if we allow multiple assignments.
            result = self.visit_single_value(value)

        # The returned value has the function's annotated result type.
        function_op = self.inserter.insertion_point.parent_op()
        assert isinstance(function_op, py.PyFuncOp)
        (result_type,) = function_op.function_type.outputs.data
        assert isinstance(result_type, py.PyObjectType)
        typed = py.PyCastOp(result, result_type)
        self.insert_op(typed, node)
        self.insert_op(py.PyReturnOp(typed.res), node)
