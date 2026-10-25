import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, i32
from xdsl.dialects.py import Py
from xdsl.frontend.pyast.lowering import PyLowering
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.frontend.pyast.utils.type_conversion import TypeConverter
from xdsl.parser import Parser


@pytest.mark.parametrize(
    "location, position",
    [('loc("test.py":2:3)', "line 2 column 2"), ("", "line 0 column 0")],
)
def test_unsupported_argument_annotation(location: str, position: str):
    # The visitor rejects these first, so they only reach the lowering directly.
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Py)
    module = Parser(
        ctx,
        f"""
        py.func @f(%x : !py.object<"float">) -> !py.object<"None"> {{
          %none = py.const None
          %result = py.cast %none : !py.object to !py.object<"None">
          py.return %result : !py.object<"None">
        }} {location}
        """,
    ).parse_module()
    type_converter = TypeConverter(globals())
    type_converter.type_registry.insert(int, i32)

    with pytest.raises(CodeGenerationException) as error:
        PyLowering(type_converter, "test.py").lower(module)
    assert str(error.value) == (
        f'Code generation exception at "test.py", {position}: '
        "Unsupported function argument type: 'float'"
    )
