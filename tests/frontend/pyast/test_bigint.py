from textwrap import dedent

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram


def test_bigint_identity() -> None:
    p = FrontendProgram()
    with CodeContext(p):

        def foo(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x

    p.compile()

    expected_result = dedent("""
    builtin.module {
      func.func @foo(%0 : bigint.bigint) -> bigint.bigint {
        func.return %0 : bigint.bigint
      }
    }
    """).strip()
    assert expected_result == p.textual_format()
