import argparse
import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from dis import dis
from pathlib import Path
from typing import Any, NamedTuple

from pysemantics import dunder_op_name, type_name

from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp
from xdsl.dialects.py import (
    CallOp,
    ConstantOp,
    ConstantValue,
    FuncOp,
    ObjectType,
    PassOp,
    ReturnOp,
)
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue


@dataclass
class OpInserter:
    """
    Class responsible for inserting operations at the right place in the
    generated IR.
    """

    insertion_point: Block
    """
    Insertion point, i.e. the pointer to the block to which the operations are
    appended.
    """

    stack: list[SSAValue] = field(default_factory=list[SSAValue])
    """
    Stack to hold the intermediate results of operations. For each new
    operation, its operands will be popped from the stack.
    """

    def get_operand(self) -> SSAValue:
        """
        Pops the last value from the operand stack and returns it.
        """
        if len(self.stack) == 0:
            raise FrontendProgramException(
                "Trying to get an operand from an empty stack."
            )
        return self.stack.pop()

    def insert_op(self, op: Operation) -> None:
        """Inserts a new operation and places its results on the stack."""
        self.insertion_point.add_op(op)
        for result in op.results:
            self.stack.append(result)

    def set_insertion_point_from_op(self, op: Operation) -> None:
        """
        Sets the insertion point to the last block in the last region of the
        operation.
        """
        if not op.regions:
            raise FrontendProgramException(
                f"Trying to set the insertion point for operation '{op.name}' with no regions."
            )
        if (last_block := op.regions[-1].blocks.last) is None:
            raise FrontendProgramException(
                f"Trying to set the insertion point for operation '{op.name}' with no blocks in its last region."
            )
        self.insertion_point = last_block

    def set_insertion_point_from_region(self, region: Region) -> None:
        """Sets the insertion point to the last block in this region."""
        if (last_block := region.blocks.last) is None:
            raise FrontendProgramException(
                "Trying to set the insertion point from the region without blocks."
            )
        self.insertion_point = last_block

    def set_insertion_point_from_block(self, block: Block) -> None:
        """Sets the insertion point to this block."""
        self.insertion_point = block


class Symbol(NamedTuple):
    name: str
    type: ObjectType
    ssa: SSAValue


class PySymbolTable:
    def __init__(self) -> None:
        self._symbols: dict[str, Symbol] = dict()

    def symbol_present(self, symbol_name: str):
        return symbol_name in self._symbols

    def symbol_need_presence(self, symbol_name: str):
        if not self.symbol_present(symbol_name):
            raise Exception(f"No such symbol {symbol_name}")

    def get_symbol(self, symbol_name: str):
        self.symbol_need_presence(symbol_name)

    def add_symbol(self, symbol: Symbol):
        self._symbols[symbol.name] = symbol


@dataclass(init=False)
class PyBuilder(ast.NodeVisitor):
    def __init__(
        self,
        module: ModuleOp,
    ):
        super().__init__()
        self.module = module

        assert len(module.body.blocks) == 1
        self.inserter = OpInserter(module.body.block)

        self.symbol_table: dict = dict()

    def gen_py(self, program: ast.AST):
        self.visit(program)
        return self.module

    def visit_Module(self, node: ast.Module):
        self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        arg_names, arg_types = extract_arguments(node.args)
        result_type = extract_return(node.returns)

        ftype = FunctionType(
            ArrayAttr(arg_types),
            ArrayAttr([result_type]),
        )

        body = Block(arg_types=ftype.inputs)
        fundef = FuncOp(
            name=node.name, arg_names=arg_names, ftype=ftype, region=Region(body)
        )

        self.inserter.insert_op(fundef)
        self.inserter.set_insertion_point_from_block(body)

        for arg_name, arg_ssa in zip(arg_names, fundef.body.first_block.args):
            self.symbol_table[arg_name] = arg_ssa

        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if isinstance(node.left, ast.Name):
            name = self.visit(node.left)
            lhs = self.symbol_table[name]
        else:
            self.visit(node.left)
            lhs = self.inserter.get_operand()

        if isinstance(node.right, ast.Name):
            name = self.visit(node.right)
            rhs = self.symbol_table[name]
        else:
            self.visit(node.right)
            rhs = self.inserter.get_operand()

        dunder = dunder_op_name(node.op)

        result_type = lhs.type if (lhs.type == rhs.type) else ObjectType("Unknown")

        self.inserter.insert_op(CallOp(dunder, [lhs, rhs], [result_type]))

    def visit_Name(self, node: ast.Name):
        return extract_Name_id(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        self.inserter.insert_op(
            ConstantOp(
                value=ConstantValue(node.value),
                result_type=ObjectType(type_name(node.value)),
            )
        )

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            self.inserter.insert_op(ReturnOp())
        else:
            self.visit(node.value)
            return_op = ReturnOp(self.inserter.get_operand())
            self.inserter.insert_op(return_op)

    def visit_Pass(self, node: ast.Pass):
        self.inserter.insert_op(PassOp())

    def visit_Call(self, node: ast.Call):

        args: Sequence[SSAValue[Attribute]] = []
        for arg in node.args:
            self.visit(arg)
            args.append(self.inserter.get_operand())

        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("")

        name = extract_Name_id(node.func)

        # Call(expr func, expr* args, keyword* keywords)
        self.inserter.insert_op(CallOp(name, args, [ObjectType("Unknown")]))


def extract_arguments(node: ast.arguments) -> tuple[list[str], list[ObjectType]]:
    return extract_args(node.args)


def extract_args(args: list[ast.arg]) -> tuple[list[str], list[ObjectType]]:
    arg_names: Sequence[str] = []
    arg_types: Sequence[ObjectType] = []

    for arg in args:
        name, arg_type = extract_arg(arg)
        arg_names.append(name)
        arg_types.append(arg_type)

    return arg_names, arg_types


def extract_arg(node: ast.arg) -> tuple[str, ObjectType]:
    if not isinstance(node.annotation, ast.Name):
        raise NotImplementedError("Only Name annotations are supported")

    type_annotation = extract_Name_id(node.annotation)
    if type_annotation == "":
        raise NotImplementedError("Arguments needs a type annotation")

    return node.arg, ObjectType(type_annotation)


def extract_return(node: ast.expr | None) -> ObjectType:
    if not isinstance(node, ast.Name):
        raise NotImplementedError("Only Name returns are supported")

    return ObjectType(extract_Name_id(node))


def extract_Name_id(node: ast.Name) -> str:
    return node.id


@dataclass(init=False)
class ASTBuilder:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Toy file")
    parser.add_argument("source", type=Path, help="toy source file")
    parser.add_argument(
        "-a",
        dest="all",
        action="store_true",
        help="More text please",
    )
    parser.add_argument(
        "-s",
        dest="source",
        action="store_true",
        help="More text please",
    )
    parser.add_argument(
        "-d",
        dest="disassemble",
        action="store_true",
        help="More text please",
    )
    parser.add_argument(
        "-t",
        dest="tree",
        action="store_true",
        help="More text please",
    )

    args = parser.parse_args()

    with open(args.source) as f:
        source = f.read()

    if args.source or args.all:
        print(f"-=-=-=-=-=[ Source ]=-=-=-=-=-\n{source}\n")

    if args.disassemble or args.all:
        print("-=-=-=-=-=[ Disassemble ]=-=-=-=-=-\n")
        dis(source)
        print()

    program = ast.parse(source)
    if args.tree or args.all:
        print(f"-=-=-=-=-=[ AST ]=-=-=-=-=-\n{ast.dump(program, indent=4)}\n")

    module = PyBuilder(ModuleOp(Region(Block()))).gen_py(program)
    print(f"-=-=-=-=-=[ Module ]=-=-=-=-=-\n{module}\n")
