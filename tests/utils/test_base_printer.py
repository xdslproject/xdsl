from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper

import pytest

from xdsl.utils.base_printer import BasePrinter


def test_indented():
    output = StringIO()
    printer = BasePrinter(stream=output)
    printer.print_string("\n{")
    with printer.indented():
        printer.print_string("\nhello\nhow are you?")
        printer.print_string("\n(")
        with printer.indented():
            printer.print_string("\nfoo,")
            printer.print_string("\nbar,")
            printer.print_string("\n")
            printer.print_string("test\nraw print!", indent=0)
            printer.print_string("\ndifferent indent level", indent=4)
        printer.print_string("\n)")
    printer.print_string("\n}")
    printer.print_string("\n[")
    with printer.indented(amount=3):
        printer.print_string("\nbaz")
    printer.print_string("\n]\n")

    EXPECTED = """
{
  hello
  how are you?
  (
    foo,
    bar,
    test
raw print!
        different indent level
  )
}
[
      baz
]
"""

    assert output.getvalue() == EXPECTED


def test_native_bytes(monkeypatch: pytest.MonkeyPatch):
    # An explicit binary stream must work even when stdout has no byte buffer.
    monkeypatch.setattr("sys.stdout", StringIO())
    output = BytesIO()
    printer = BasePrinter(output, content_type=bytes)
    printer.print_string("é")
    with printer.indented():
        printer.print_content(b"\n\xff\x00\nend")
        printer.print_content(b"\nraw\n\x80", indent=0)
    assert output.getvalue() == b"\xc3\xa9\n  \xff\x00\n  end\nraw\n\x80"


@pytest.mark.parametrize("binary", [False, True])
def test_default_stream_is_bound_at_construction(binary: bool):
    original_output, redirected_output = BytesIO(), BytesIO()
    with (
        TextIOWrapper(original_output, encoding="utf-8") as original_stdout,
        TextIOWrapper(redirected_output, encoding="utf-8") as redirected_stdout,
    ):
        with redirect_stdout(original_stdout):
            printer = BasePrinter(content_type=bytes) if binary else BasePrinter()
        with redirect_stdout(redirected_stdout):
            printer.print_string("é\n")

        original_stdout.flush()
        redirected_stdout.flush()
        assert original_output.getvalue() == b"\xc3\xa9\n"
        assert redirected_output.getvalue() == b""
