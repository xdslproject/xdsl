import ast
import xdsl.dialects.builtin as builtin
import xdsl.dialects.func as func
import xdsl.frontend.symref as symref

from dataclasses import dataclass, field
from typing import Any, Dict, List
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.op_inserter import OpInserter
from xdsl.frontend.op_resolver import OpResolver
from xdsl.frontend.type_conversion import TypeConverter
from xdsl.ir import Attribute, Block, Region


@dataclass
class CodeGeneration:

    @staticmethod
    def run_with_type_converter(type_converter: TypeConverter,
                                stmts: List[ast.stmt]) -> builtin.ModuleOp:
        """Generates xDSL code and returns it encapsulated into a single module."""
        module = builtin.ModuleOp.from_region_or_ops([])
        visitor = CodegGenerationVisitor(type_converter, module)
        for stmt in stmts:
            visitor.visit(stmt)
        return module


@dataclass
class CodegGenerationVisitor(ast.NodeVisitor):
    """Visitor that generates xDSL from the Python AST."""

    type_converter: TypeConverter = field(init=False)
    """Used for type conversion during code generation."""

    globals: Dict[str, Any] = field(init=False)
    """
    Imports and other global information from the module, useful for looking
    up classes, etc.
    """

    inserter: OpInserter = field(init=False)
    """Used for inserting newly generated operations to the right block."""

    symbol_table: Dict[str, Attribute] | None = field(default=None)
    """
    Maps local variable names to their xDSL types. A single dictionary is sufficient
    because inner functions and global variables are not allowed (yet).
    """

    def __init__(self, type_converter: TypeConverter,
                 module: builtin.ModuleOp) -> None:
        self.type_converter = type_converter
        self.globals = type_converter.globals

        assert len(module.body.blocks) == 1
        self.inserter = OpInserter(module.body.blocks[0])

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        raise CodeGenerationException(
            node.lineno, node.col_offset,
            f"Unsupported Python AST node {str(node)}")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # TODO: Implement assignemnt in the next patch.
        pass

    def visit_Assign(self, node: ast.Assign) -> None:
        # TODO: Implement assignemnt in the next patch.
        pass

    def visit_BinOp(self, node: ast.BinOp):
        op_name: str = node.op.__class__.__name__

        # Local table which maps the name of a binry operator to the
        # corresponding Python function.
        op_to_python = {
            "Add": "__add__",
            "BitAnd": "__and__",
            "LShift": "__lshift__",
            "Mult": "__mul__",
            "RShift": "__rshift__",
            "Sub": "__sub__",
        }
        if op_name not in op_to_python:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Unknown binary operation {op_name}.")

        self.visit(node.right)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()
        if lhs.typ != rhs.typ:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Expected the same types for binary operation '{op_name}', "
                f"but got {lhs.typ} and {rhs.typ}.")

        # Look-up what is the frontend type we deal with to resolve the binary
        # operation.
        frontend_type = self.type_converter.xdsl_to_frontend_type_map[
            lhs.typ.__class__]

        op = OpResolver.resolve_op_overload(op_to_python[op_name],
                                            frontend_type)(lhs, rhs)
        self.inserter.insert_op(op)

    def visit_Compare(self, node: ast.Compare):
        # Allow a single comparison only.
        if len(node.comparators) != 1 or len(node.ops) != 1:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Expected a single comparator, but found {len(node.comparators)}.")
        comp = node.comparators[0]
        op_name: str = node.ops[0].__class__.__name__

        # Local table which maps the name of a comparison operator to the
        # corresponding Python functions and xDSL mnemonics.
        op_to_python_and_mnemonic = {
            "Eq": ("__eq__", "eq"),
            "Gt": ("__gt__", "sgt"),
            "GtE": ("__ge__", "sge"),
            "Lt": ("__lt__", "slt"),
            "LtE": ("__le__", "sle"),
            "NotEq": ("__ne__", "ne"),
        }
        if op_name not in op_to_python_and_mnemonic:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Unknown comparison operation '{op_name}'.")
  
        self.visit(comp)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()
        if lhs.typ != rhs.typ:
            raise CodeGenerationException(
                node.lineno, node.col_offset,
                f"Expected the same types for comparison operator '{op_name}',"
                f" but got {lhs.typ} and {rhs.typ}.")

        python_op = op_to_python_and_mnemonic[op_name][0]
        mnemonic = op_to_python_and_mnemonic[op_name][1]
        frontend_type = self.type_converter.xdsl_to_frontend_type_map[
            lhs.typ.__class__]

        op = OpResolver.resolve_op_overload(python_op, frontend_type)(lhs, rhs,
                                                                      mnemonic)
        self.inserter.insert_op(op)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:

        # Set the symbol table.
        assert self.symbol_table is None
        self.symbol_table = dict()

        # Then, convert types in the function signature.
        argument_types: List[Attribute] = []
        for i, arg in enumerate(node.args.args):
            if arg.annotation is None:
                raise CodeGenerationException(arg.lineno, arg.col_offset, f"")
            xdsl_type = self.type_converter.convert_type_hint(arg.annotation)
            argument_types.append(xdsl_type)

        return_types: List[Attribute] = []
        if node.returns is not None:
            xdsl_type = self.type_converter.convert_type_hint(node.returns)
            return_types.append(xdsl_type)

        # Create a function operation.
        entry_block = Block()
        body_region = Region.from_block_list([entry_block])
        func_op = func.FuncOp.from_region(node.name, argument_types,
                                          return_types, body_region)

        self.inserter.insert_op(func_op)
        self.inserter.set_insertion_point_from_block(entry_block)

        # All arguments are declared using symref.
        for i, arg in enumerate(node.args.args):
            symbol_name = str(arg.arg)
            block_arg = entry_block.insert_arg(argument_types[i], i)
            self.symbol_table[symbol_name] = argument_types[i]
            entry_block.add_op(symref.Declare.get(symbol_name))
            entry_block.add_op(symref.Update.get(symbol_name, block_arg))

        # Parse function body.
        for stmt in node.body:
            self.visit(stmt)

        # When function definition is processed, reset the symbol table and set
        # the insertion point.
        self.symbol_table = None
        parent_op = func_op.parent_op()
        assert parent_op is not None
        self.inserter.set_insertion_point_from_op(parent_op)

    def visit_Name(self, node: ast.Name):
        fetch_op = symref.Fetch.get(node.id, self.symbol_table[node.id])
        self.inserter.insert_op(fetch_op)

    def visit_Pass(self, node: ast.Pass) -> None:
        parent_op = self.inserter.insertion_point.parent_op()

        # We might have to add an explicit return statement in this case. Make sure to
        # check the type signature.
        if parent_op is not None and isinstance(parent_op, func.FuncOp):
            return_types = parent_op.function_type.outputs.data

            if len(return_types) != 0:
                function_name = parent_op.attributes["sym_name"].data
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"Expected '{function_name}' to return a type.")
            self.inserter.insert_op(func.Return.get())

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
                node.lineno, node.col_offset,
                "Return statement should be placed only at the end of the "
                "function body.")

        func_name = parent_op.attributes["sym_name"].data
        func_return_types = parent_op.function_type.outputs.data

        if node.value is None:
            # Return nothing, check function signature matches.
            if len(func_return_types) != 0:
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"Expected non-zero number of return types in function "
                    f"'{func_name}', but got 0.")
            self.inserter.insert_op(func.Return.get())
        else:
            # Return some type, check function signature matches as well.
            # TODO: Support multiple return values if we allow multiple assignemnts.
            self.visit(node.value)
            operands = [self.inserter.get_operand()]

            if len(func_return_types) == 0:
                raise CodeGenerationException(
                    node.lineno, node.col_offset,
                    f"Expected no return types in function '{func_name}'.")

            for i in range(len(operands)):
                if func_return_types[i] != operands[i].typ:
                    raise CodeGenerationException(
                        node.lineno, node.col_offset,
                        f"Type signature and the type of the return value do "
                        f"not match at position {i}: expected {func_return_types[i]},"
                        f" got {operands[i].typ}.")

            self.inserter.insert_op(func.Return.get(*operands))
