import ast

from xdsl.dialects import builtin, py
from xdsl.frontend.pyast.code_generation import CodeGenerationVisitor
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter


def test_generates_py():
    source = """
def f(x: int, y: float) -> int:
    z = g(x, key=y)
    assert z < x
    return h()
"""
    module = builtin.ModuleOp([])
    visitor = CodeGenerationVisitor(TypeConverter(), module, "test.py")
    visitor.visit(ast.parse(source).body[0])
    assert (
        str(module)
        == """\
builtin.module {
  py.func @f(%x: !py.object<"int">, %y: !py.object<"float">) -> !py.object<"int"> {
    symref.declare "x"
    %0 = py.cast %x : !py.object<"int"> to !py.object
    symref.update @x = %0 : !py.object
    symref.declare "y"
    %1 = py.cast %y : !py.object<"float"> to !py.object
    symref.update @y = %1 : !py.object
    %2 = symref.fetch @x : !py.object
    %3 = symref.fetch @y : !py.object
    %4 = py.call "g"(%2, %3) keywords ["key"]
    symref.declare "z"
    symref.update @z = %4 : !py.object
    %5 = symref.fetch @x : !py.object
    %6 = symref.fetch @z : !py.object
    %7 = py.compare lt, %6, %5
    cf.assert %7, ""
    %8 = py.call "h"()
    %9 = py.cast %8 : !py.object to !py.object<"int">
    py.return %9 : !py.object<"int">
  }
}"""
    )

    # Operations are located at the code they come from, with one-based columns.
    call = next(op for op in module.walk() if isinstance(op, py.PyCallOp))
    assert call.location == builtin.FileLineColLoc(
        builtin.StringAttr("test.py"), builtin.IntAttr(3), builtin.IntAttr(9)
    )


class StackRecordingVisitor(CodeGenerationVisitor):
    """Record the value stack size before and after each expression statement."""

    stack_sizes: list[tuple[int, int]]

    def visit_Expr(self, node: ast.Expr) -> None:
        before = len(self.inserter.stack)
        super().visit_Expr(node)
        self.stack_sizes.append((before, len(self.inserter.stack)))


def test_expression_statements_discard_their_values():
    type_converter = TypeConverter()
    type_converter.type_registry.insert(int, builtin.i32)
    source = """
def discarded(x: int) -> None:
    pair(x, x)
    consume(x)
    add(x, x)
"""
    visitor = StackRecordingVisitor(type_converter, builtin.ModuleOp([]), None)
    visitor.stack_sizes = []
    visitor.visit(ast.parse(source).body[0])

    assert len(visitor.stack_sizes) == 3
    assert all(before == after for before, after in visitor.stack_sizes)
