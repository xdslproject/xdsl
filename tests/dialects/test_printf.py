import pytest

from xdsl.dialects import builtin, test
from xdsl.dialects import printf as print_dialect
from xdsl.transforms import printf_to_llvm


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
    assert printf_to_llvm.legalize_str_for_symbol_name(given) == expected


def test_format_str_from_op():
    a1, a2 = test.TestOp.create(result_types=[builtin.i32, builtin.f32]).results
    op = print_dialect.PrintFormatOp("test {} value {}", a1, a2)

    parts = printf_to_llvm._format_string_spec_from_print_op(  # pyright: ignore[reportPrivateUsage]
        op
    )

    assert list(parts) == [
        "test ",
        a1,
        " value ",
        a2,
    ]

    op2 = print_dialect.PrintFormatOp("{}", a1)

    parts2 = printf_to_llvm._format_string_spec_from_print_op(  # pyright: ignore[reportPrivateUsage]
        op2
    )

    assert list(parts2) == [a1]


def test_global_symbol_name_generation():
    """
    Check that two strings that are invalid symbol names still result in two distinct
    global symbol names.

    Similarly, test that the same string results in the same symbol name.
    """
    s1 = printf_to_llvm._key_from_str("(")  # pyright: ignore[reportPrivateUsage]
    s2 = printf_to_llvm._key_from_str(")")  # pyright: ignore[reportPrivateUsage]

    assert s1 != s2

    s3 = printf_to_llvm._key_from_str(")")  # pyright: ignore[reportPrivateUsage]

    assert s2 == s3
