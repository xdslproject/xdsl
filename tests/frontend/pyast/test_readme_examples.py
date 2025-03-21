from textwrap import dedent

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.dialects.builtin import i32
from xdsl.frontend.pyast.program import FrontendProgram


def test_readme_example() -> None:
    p = FrontendProgram()
    with CodeContext(p):

        def foo(x: i32, y: i32, z: i32) -> i32:  # pyright: ignore[reportUnusedFunction]
            return x + y * z

    p.compile()

    expected_result = dedent("""
    builtin.module {
      func.func @foo(%0 : i32, %1 : i32, %2 : i32) -> i32 {
        %3 = arith.muli %1, %2 : i32
        %4 = arith.addi %0, %3 : i32
        func.return %4 : i32
      }
    }
    """).strip()
    assert expected_result == p.textual_format()
