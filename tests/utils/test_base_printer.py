from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import IO
from unittest.mock import Mock

import pytest

from xdsl.backend.wgsl.wgsl_printer import WGSLPrinter
from xdsl.printer import Printer
from xdsl.utils.base_printer import BasePrinter
from xdsl.utils.color_printer import ColorPrinter


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
    printer = BasePrinter(output)
    printer.print_string("é")
    with printer.indented():
        printer.print_content(b"\n\xff\x00\nend")
        printer.print_content(b"\nraw\n\x80", indent=0)
    assert output.getvalue() == b"\xc3\xa9\n  \xff\x00\n  end\nraw\n\x80"


def test_custom_encoding():
    output = BytesIO()
    printer = BasePrinter(
        output, encode=lambda text: text.encode("latin-1"), indent_num_spaces=4
    )
    with printer.indented():
        printer.print_string("\né")
        printer.print_content(b"\xff")
    assert output.getvalue() == b"\n    \xe9\xff"


@pytest.mark.parametrize("buffering", [0, -1])
def test_binary_file(tmp_path: Path, buffering: int):
    path = tmp_path / "output"
    with path.open("wb", buffering=buffering) as stream:
        printer = BasePrinter(stream)
        printer.print_string("é\n")
    assert path.read_bytes() == b"\xc3\xa9\n"


def test_custom_stream_requires_encoding():
    stream: IO[str] = Mock(spec=IO)
    with pytest.raises(TypeError, match="encode must be supplied for custom streams"):
        BasePrinter(stream)


def test_custom_stream_with_encoding():
    stream = Mock(spec=IO)
    printer = BasePrinter[str](stream, encode=str)
    printer.print_string("é\n")
    stream.write.assert_called_once_with("é\n")


@pytest.mark.parametrize(
    "printer_type", [BasePrinter, ColorPrinter, Printer, WGSLPrinter]
)
def test_default_stream_is_bound_at_construction(
    printer_type: type[BasePrinter[str]],
):
    original_output, redirected_output = BytesIO(), BytesIO()
    with (
        TextIOWrapper(original_output, encoding="utf-8") as original_stdout,
        TextIOWrapper(redirected_output, encoding="utf-8") as redirected_stdout,
    ):
        with redirect_stdout(original_stdout):
            printer = printer_type()
        with redirect_stdout(redirected_stdout):
            printer.print_string("é\n")

        original_stdout.flush()
        redirected_stdout.flush()
        assert original_output.getvalue() == b"\xc3\xa9\n"
        assert redirected_output.getvalue() == b""
