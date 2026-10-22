import ast
from textwrap import dedent

from xdsl.dialects import builtin, py
from xdsl.frontend.pyast.code_generation import CodeGenerationVisitor


def generate(source: str) -> builtin.ModuleOp:
    module = builtin.ModuleOp([])
    function = ast.parse(dedent(source)).body[0]
    CodeGenerationVisitor(module, "test.py").visit(function)
    return module


def test_generates_py():
    module = generate("""
    def f(x: int, y: float) -> int:
        z = g(x, key=1.5)
        assert z < x
        return z if x else h()
    """)
    assert str(module) == dedent("""\
        builtin.module {
          py.func @f ["int", "float"] -> "int" {
          ^bb0(%x: !py.object, %y: !py.object):
            symref.declare "x"
            symref.update @x = %x : !py.object
            symref.declare "y"
            symref.update @y = %y : !py.object
            %0 = symref.fetch @x : !py.object
            %1 = py.const 1.5
            %2 = py.call "g"(%0, %1) keywords ["key"]
            symref.declare "z"
            symref.update @z = %2 : !py.object
            %3 = symref.fetch @x : !py.object
            %4 = symref.fetch @z : !py.object
            %5 = py.compare "lt" %4 %3
            %6 = py.bool %5
            cf.assert %6, ""
            %7 = symref.fetch @x : !py.object
            %8 = py.bool %7
            %9 = scf.if %8 -> (!py.object) {
              %10 = symref.fetch @z : !py.object
              scf.yield %10 : !py.object
            } else {
              %11 = py.call "h"()
              scf.yield %11 : !py.object
            }
            py.return %9
          }
        }""")

    # Operations and arguments are located at the code they come from, with
    # one-based columns.
    function = module.body.block.first_op
    assert isinstance(function, py.PyFuncOp)
    assert function.location == builtin.FileLineColLoc(
        builtin.StringAttr("test.py"), builtin.IntAttr(2), builtin.IntAttr(1)
    )
    assert function.body.block.args[1].location == builtin.FileLineColLoc(
        builtin.StringAttr("test.py"), builtin.IntAttr(2), builtin.IntAttr(15)
    )
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
    source = """
def discarded(x: int) -> None:
    pair(x, x)
    consume(x)
    x
"""
    visitor = StackRecordingVisitor(builtin.ModuleOp([]), None)
    visitor.stack_sizes = []
    visitor.visit(ast.parse(source).body[0])

    assert len(visitor.stack_sizes) == 3
    assert all(before == after for before, after in visitor.stack_sizes)
