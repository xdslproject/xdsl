import ast
from io import StringIO

from xdsl.dialects import arith, builtin, test
from xdsl.frontend.pyast.code_generation import CodeGenerationVisitor
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.ir import SSAValue
from xdsl.printer import Printer


def add(operand1: int, operand2: int) -> int:
    return operand1 + operand2


def pair(lhs: int, rhs: int) -> tuple[int, bool]:
    return lhs + rhs, False


def consume(value: int) -> None:
    pass


def build_consume(value: SSAValue) -> test.TestOp:
    return test.TestOp([value])


class StackRecordingVisitor(CodeGenerationVisitor):
    """Record the value stack size before and after each expression statement."""

    stack_sizes: list[tuple[int, int]]

    def visit_Expr(self, node: ast.Expr) -> None:
        before = len(self.inserter.stack)
        super().visit_Expr(node)
        self.stack_sizes.append((before, len(self.inserter.stack)))


def test_expression_statements_discard_their_values():
    type_converter = TypeConverter(globals())
    type_converter.type_registry.insert(int, builtin.i32)
    type_converter.function_registry.insert(add, arith.AddiOp)
    type_converter.function_registry.insert(pair, arith.AddUIExtendedOp)
    type_converter.function_registry.insert(consume, build_consume)
    source = """
def discarded(x: int) -> None:
    pair(x, x)
    consume(x)
    add(x, x)
"""
    visitor = StackRecordingVisitor(type_converter, builtin.ModuleOp([]), None)
    visitor.stack_sizes = []
    visitor.visit(ast.parse(source).body[0])

    # Calls with two, zero and one results all leave the stack as they found it.
    assert len(visitor.stack_sizes) == 3
    assert all(before == after for before, after in visitor.stack_sizes)


def test_operations_are_located_at_their_source():
    type_converter = TypeConverter(globals())
    type_converter.type_registry.insert(int, builtin.i32)
    type_converter.function_registry.insert(add, arith.AddiOp)
    source = """
def f(x: int) -> int:
    y = add(x, x)
    return y
"""
    module = builtin.ModuleOp([])
    visitor = CodeGenerationVisitor(type_converter, module, "test.py")
    visitor.visit(ast.parse(source).body[0])

    stream = StringIO()
    Printer(stream, print_debuginfo=True).print_op(module)
    assert (
        stream.getvalue()
        == """builtin.module {
  func.func @f(%x: i32 loc(unknown)) -> i32 {
    symref.declare "x" loc("test.py":2:7)
    symref.update @x = %x : i32 loc("test.py":2:7)
    %0 = symref.fetch @x : i32 loc("test.py":3:13)
    %1 = symref.fetch @x : i32 loc("test.py":3:16)
    %2 = arith.addi %0, %1 : i32 loc("test.py":3:9)
    symref.declare "y" loc("test.py":3:5)
    symref.update @y = %2 : i32 loc("test.py":3:5)
    %3 = symref.fetch @y : i32 loc("test.py":4:12)
    func.return %3 : i32 loc(unknown)
  } loc(unknown)
} loc(unknown)"""
    )
