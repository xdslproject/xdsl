from io import StringIO

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
