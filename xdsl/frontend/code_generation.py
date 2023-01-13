import ast
import xdsl.dialects.builtin as builtin
import xdsl.dialects.func as func
import xdsl.frontend.symref as symref

from dataclasses import dataclass, field
from typing import Any, Dict, List
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.op_inserter import OpInserter
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
