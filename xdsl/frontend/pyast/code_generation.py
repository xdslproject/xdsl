import ast
from dataclasses import dataclass, field
from typing import Any, cast

import xdsl.dialects.affine as affine
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.cf as cf
import xdsl.dialects.func as func
import xdsl.dialects.scf as scf
import xdsl.frontend.pyast.symref as symref
from xdsl.frontend.pyast.exception import (
    CodeGenerationException,
)
from xdsl.frontend.pyast.op_inserter import OpInserter
from xdsl.frontend.pyast.python_code_check import FunctionMap
from xdsl.frontend.pyast.type_conversion import TypeConverter
from xdsl.ir import Attribute, Block, Region, SSAValue, TypeAttribute


@dataclass
class CodeGeneration:
    @staticmethod
    def run_with_type_converter(
        type_converter: TypeConverter,
        functions_and_blocks: FunctionMap,
        file: str | None,
    ) -> builtin.ModuleOp:
        """Generates xDSL code and returns it encapsulated into a single module."""
        module = builtin.ModuleOp([])

        visitor = CodeGenerationVisitor(type_converter, module, file)
        for function_def, _ in functions_and_blocks.values():
            visitor.visit(function_def)
        return module


@dataclass(init=False)
class CodeGenerationVisitor(ast.NodeVisitor):
    """Visitor that generates xDSL from the Python AST."""

    type_converter: TypeConverter
    """Used for type conversion during code generation."""

    inserter: OpInserter
    """Used for inserting newly generated operations to the right block."""

    symbol_table: dict[str, Attribute] | None = field(default=None)
    """
    Maps local variable names to their xDSL types. A single dictionary is sufficient
    because inner functions and global variables are not allowed (yet).
    """

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

    def generic_visit(self, node: ast.AST) -> None:
        raise CodeGenerationException(
            self.file,
            getattr(node, "lineno"),
            getattr(node, "col_offset"),
            f"Unsupported Python AST node {str(node)}",
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # TODO: Implement assignemnt in the next patch.
        pass

    def visit_Assert(self, node: ast.Assert):
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
        op = cf.AssertOp(self.inserter.get_operand(), msg)
        self.inserter.insert_op(op)

    def visit_Assign(self, node: ast.Assign) -> None:
        # TODO: Implement assignemnt in the next patch.
        pass

    def visit_BinOp(self, node: ast.BinOp):
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
            self.inserter.insert_op(op)
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

    def visit_Call(self, node: ast.Call):
        # Resolve function
        assert isinstance(node.func, ast.Name)
        func_name = node.func.id
        source_func = self.type_converter.globals.get(func_name, None)
        if source_func is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Function '{func_name}' is not defined in scope.",
            )
        ir_op = self.type_converter.function_registry.get_operation_type(source_func)
        if ir_op is None:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Function '{func_name}' is not registered.",
            )

        # Resolve arguments
        assert self.symbol_table is not None
        args: list[symref.FetchOp] = []
        for arg in node.args:
            if not isinstance(arg, ast.Name) or arg.id not in self.symbol_table:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Function arguments must be declared variables.",
                )
            args.append(arg_op := symref.FetchOp(arg.id, self.symbol_table[arg.id]))
            self.inserter.insert_op(arg_op)

        # Resolve keyword arguments
        kwargs: dict[str, symref.FetchOp] = {}
        for keyword in node.keywords:
            if (
                not isinstance(keyword.value, ast.Name)
                or keyword.value.id not in self.symbol_table
            ):
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    "Function arguments must be declared variables.",
                )
            assert keyword.arg is not None
            kwargs[keyword.arg] = symref.FetchOp(
                keyword.value.id, self.symbol_table[keyword.value.id]
            )
            self.inserter.insert_op(kwargs[keyword.arg])

        self.inserter.insert_op(ir_op(*args, **kwargs))

    def visit_Compare(self, node: ast.Compare):
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
            self.inserter.insert_op(op)
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

    def _generate_affine_loop_bounds(
        self, args: list[ast.expr]
    ) -> tuple[int, int, int]:
        # Process loop start.
        if len(args) <= 1:
            start = 0
        else:
            assert isinstance(args[0], ast.Constant)
            if not isinstance(args[0].value, int):
                raise CodeGenerationException(
                    self.file,
                    args[0].lineno,
                    args[0].col_offset,
                    f"Expected integer constant for loop start, got '{type(args[0].value).__qualname__}'.",
                )
            start = int(args[0].value)

        # Process loop end.
        if len(args) <= 1:
            arg = args[0]
        else:
            arg = args[1]
        assert isinstance(arg, ast.Constant)
        if not isinstance(arg.value, int):
            raise CodeGenerationException(
                self.file,
                arg.lineno,
                arg.col_offset,
                f"Expected integer constant for loop end, got '{type(arg.value).__qualname__}'.",
            )
        end = int(arg.value)

        # Process loop step.
        if len(args) == 3:
            assert isinstance(args[2], ast.Constant)
            if not isinstance(args[2].value, int):
                raise CodeGenerationException(
                    self.file,
                    args[2].lineno,
                    args[2].col_offset,
                    f"Expected integer constant for loop step, got '{type(args[2].value).__qualname__}'.",
                )
            step = int(args[2].value)
        else:
            step = 1

        return start, end, step

    def _generate_loop_bounds(
        self, args: list[ast.expr]
    ) -> tuple[SSAValue, SSAValue, SSAValue]:
        # Process loop start.
        if len(args) <= 1:
            start = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
            self.inserter.insert_op(start)
        else:
            self.visit(args[0])
        start = self.inserter.get_operand()
        if not isinstance(start.type, builtin.IndexType):
            raise CodeGenerationException(
                self.file,
                args[0].lineno,
                args[0].col_offset,
                f"Expected 'index' type for loop start, got '{start.type}'.",
            )

        # Process loop end.
        if len(args) <= 1:
            arg = args[0]
        else:
            arg = args[1]
        self.visit(arg)
        end = self.inserter.get_operand()
        if not isinstance(end.type, builtin.IndexType):
            raise CodeGenerationException(
                self.file,
                arg.lineno,
                arg.col_offset,
                f"Expected 'index' type for loop end, got '{end.type}'.",
            )

        # Process loop step.
        if len(args) == 3:
            self.visit(args[2])
        else:
            step = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
            self.inserter.insert_op(step)
        step = self.inserter.get_operand()
        if not isinstance(step.type, builtin.IndexType):
            raise CodeGenerationException(
                self.file,
                args[2].lineno,
                args[2].col_offset,
                f"Expected 'index' type for loop step, got '{step.type}'.",
            )

        return start, end, step

    def visit_For(self, node: ast.For):
        # Let's not support weird Python syntax right now.
        if len(node.orelse) > 0:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Else clause in for loops is not supported.",
            )

        # Loops in Python can iterate over multiple variables, using enumerate,
        # zip, range, and other builtin features. Since we cannot support
        # everything immediately, the general idea is to support basic
        # range-based loops only.
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Only range-based loops are supported.",
            )

        if len(node.iter.args) < 1 or 3 < len(node.iter.args):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Range-based loop expected between 1 and 3 arguments, but got "
                f"{len(node.iter.args)}.",
            )
        if not isinstance(node.target, ast.Name):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Range-based loop must take a single target variable, e.g. `for"
                " i in range(10)`.",
            )

        # Now that all checks passed, we can proceed with code generation.

        # If iteration space is constant, generate affine for loop. Note that
        # there are corner cases, e.g. step = -1 which this check will treat as
        # non-affine due to minus being a unary operator.
        assert isinstance(node.iter, ast.Call)
        is_affine = all([isinstance(arg, ast.Constant) for arg in node.iter.args])

        # Add target vaariable as an entry block argument.
        assert isinstance(node.target, ast.Name)
        body = Region([Block(arg_types=(builtin.IndexType(),))])

        # Process loop bounds.
        if is_affine:
            start, end, step = self._generate_affine_loop_bounds(node.iter.args)
        else:
            start, end, step = self._generate_loop_bounds(node.iter.args)

        # Generate loop body and make sure we have a terminator. It is the
        # responsibility of the subsequent passes of the front-end to perform
        # SSA-construction and yield appropriate operands.
        curr_block = self.inserter.insertion_point
        self.inserter.set_insertion_point_from_region(body)
        for stmt in node.body:
            self.visit(stmt)
        if is_affine:
            self.inserter.insert_op(affine.YieldOp.get())
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(step, int)
            op = affine.ForOp.from_region([], [], [], [], start, end, body, step)
        else:
            self.inserter.insert_op(scf.YieldOp())
            assert isinstance(start, SSAValue)
            assert isinstance(end, SSAValue)
            assert isinstance(step, SSAValue)
            op = scf.ForOp(start, end, step, [], body)

        self.inserter.set_insertion_point_from_block(curr_block)
        self.inserter.insert_op(op)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Set the symbol table.
        assert self.symbol_table is None
        self.symbol_table = dict()

        # Then, convert types in the function signature.
        argument_types: list[Attribute] = []
        for i, arg in enumerate(node.args.args):
            if arg.annotation is None:
                raise CodeGenerationException(
                    self.file,
                    arg.lineno,
                    arg.col_offset,
                    "Function arguments must be type hinted",
                )
            xdsl_type = self.type_converter.type_registry.resolve_attribute(
                ast.unparse(arg.annotation), self.type_converter.globals
            )
            if xdsl_type is None:
                raise CodeGenerationException(
                    self.file,
                    arg.lineno,
                    arg.col_offset,
                    f"Unsupported function argument type: '{ast.unparse(arg.annotation)}'",
                )
            argument_types.append(xdsl_type)

        return_types: list[Attribute] = []
        if node.returns is not None:
            xdsl_type = self.type_converter.type_registry.resolve_attribute(
                ast.unparse(node.returns), self.type_converter.globals
            )
            if xdsl_type is None:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    f"Unsupported function return type: '{ast.unparse(node.returns)}'",
                )
            return_types.append(xdsl_type)

        # Create a function operation.
        entry_block = Block()
        body_region = Region(entry_block)
        func_op = func.FuncOp.from_region(
            node.name, argument_types, return_types, body_region
        )

        self.inserter.insert_op(func_op)
        self.inserter.set_insertion_point_from_block(entry_block)

        # All arguments are declared using symref.
        for i, arg in enumerate(node.args.args):
            symbol_name = str(arg.arg)
            block_arg = entry_block.insert_arg(argument_types[i], i)
            block_arg.name_hint = symbol_name
            self.symbol_table[symbol_name] = argument_types[i]
            entry_block.add_op(symref.DeclareOp(symbol_name))
            entry_block.add_op(symref.UpdateOp(symbol_name, block_arg))

        # Parse function body.
        for stmt in node.body:
            self.visit(stmt)

        # When function definition is processed, reset the symbol table and set
        # the insertion point.
        self.symbol_table = None
        parent_op = func_op.parent_op()
        assert parent_op is not None
        self.inserter.set_insertion_point_from_op(parent_op)

    def visit_If(self, node: ast.If):
        # Get the condition.
        self.visit(node.test)
        cond = self.inserter.get_operand()
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
        self.inserter.insert_op(op)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.visit(node.test)
        cond = self.inserter.get_operand()
        cond_block = self.inserter.insertion_point

        def visit_expr(expr: ast.expr) -> tuple[Attribute, Region]:
            region = Region([Block()])
            self.inserter.set_insertion_point_from_region(region)
            self.visit(expr)
            result = self.inserter.get_operand()
            self.inserter.insert_op(scf.YieldOp(result))
            return result.type, region

        # Generate code for both branches.
        true_type, true_region = visit_expr(node.body)
        false_type, false_region = visit_expr(node.orelse)

        # Check types are the same for this to be a valid if statement.
        if true_type != false_type:
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                f"Expected the same types for if expression,"
                f" but got {true_type} and {false_type}.",
            )
        op = scf.IfOp(cond, [true_type], true_region, false_region)

        # Reset insertion point to add scf.if.
        self.inserter.set_insertion_point_from_block(cond_block)
        self.inserter.insert_op(op)

    def visit_Name(self, node: ast.Name):
        fetch_op = symref.FetchOp(node.id, self.get_symbol(node))
        self.inserter.insert_op(fetch_op)

    def visit_Pass(self, node: ast.Pass) -> None:
        parent_op = self.inserter.insertion_point.parent_op()

        # We might have to add an explicit return statement in this case. Make sure to
        # check the type signature.
        if parent_op is not None and isinstance(parent_op, func.FuncOp):
            return_types = parent_op.function_type.outputs.data

            if len(return_types) != 0:
                function_name = parent_op.sym_name.data
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    f"Expected '{function_name}' to return a type.",
                )
            self.inserter.insert_op(func.ReturnOp())

    def visit_Return(self, node: ast.Return) -> None:
        # First of all, we should only be able to return if the statement is directly
        # in the function. Cases like:
        #
        # def foo(cond: i1):
        #   if cond:
        #     return 1
        #   else:
        #     return 0
        #
        # are not allowed at the moment.
        parent_op = self.inserter.insertion_point.parent_op()
        if not isinstance(parent_op, func.FuncOp):
            raise CodeGenerationException(
                self.file,
                node.lineno,
                node.col_offset,
                "Return statement should be placed only at the end of the "
                "function body.",
            )

        callee = parent_op.sym_name.data
        func_return_types = parent_op.function_type.outputs.data

        if node.value is None:
            # Return nothing, check function signature matches.
            if len(func_return_types) != 0:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    f"Expected non-zero number of return types in function "
                    f"'{callee}', but got 0.",
                )
            self.inserter.insert_op(func.ReturnOp())
        else:
            # Return some type, check function signature matches as well.
            # TODO: Support multiple return values if we allow multiple assignemnts.
            self.visit(node.value)
            operands = [self.inserter.get_operand()]

            if len(func_return_types) == 0:
                raise CodeGenerationException(
                    self.file,
                    node.lineno,
                    node.col_offset,
                    f"Expected no return types in function '{callee}'.",
                )

            for i in range(len(operands)):
                if func_return_types[i] != operands[i].type:
                    raise CodeGenerationException(
                        self.file,
                        node.lineno,
                        node.col_offset,
                        f"Type signature and the type of the return value do "
                        f"not match at position {i}: expected {func_return_types[i]},"
                        f" got {operands[i].type}.",
                    )

            self.inserter.insert_op(func.ReturnOp(*operands))
