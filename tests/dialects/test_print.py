from xdsl.dialects import print, test, builtin
from xdsl.transforms import print_to_println
import pytest


@pytest.mark.parametrize(
    "given,expected",
    (
        ("test", "test"),
        ("@hello  world", "hello_world"),
        ("something.with.dots", "something_with_dots"),
        ("this is 💩", "this_is"),
        ("123 is a number!", "123_is_a_number"),
    ),
)
def test_symbol_sanitizer(given: str, expected: str):
    assert print_to_println.legalize_str_for_symbol_name(given) == expected


def test_format_str_from_op():
    a1, a2 = test.TestOp.create(result_types=[builtin.i32, builtin.f32]).results
    op = print.PrintLnOp("test {} value {}", a1, a2)

    parts = print_to_println._format_string_spec_from_print_op(op)  # type: ignore

    assert list(parts) == [
        "test ",
        a1,
        " value ",
        a2,
    ]
