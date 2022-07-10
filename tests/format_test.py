from __future__ import annotations

import sys
from io import StringIO
from typing import List

from xdsl.error_format import Frame
from xdsl.ir import Data
from xdsl.irdl import VerifyException, irdl_attr_definition
from xdsl.parser import Parser
from xdsl.printer import Printer

# using classes from other modules


@irdl_attr_definition
class IntListData(Data[List[int]]):
    """
    An attribute holding a list of integers.
    """
    name = "int_list"

    @staticmethod
    def parse_parameter(parser: Parser) -> List[int]:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: List[int], printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(data, lambda x: printer.print_string(str(x)))
        printer.print_string("]")

    def verify(self) -> None:
        if not isinstance(self.data, list):
            raise VerifyException("int_list data should hold a list.")
        for elem in self.data:
            if not isinstance(elem, int):
                raise VerifyException(
                    "int_list list elements should be integers.")


def test_simple_data_constructor_failure():
    """
    Test that the verifier of a Data with a non-class parameter fails when
    given wrong arguments.
    """
    try:
        IntListData([0, 1, 42, ""])  # type: ignore

    except VerifyException as e:
        simple_test(e)


def simple_test(e):
    x = Frame(e)
    gen = list(x.get_frame(1))
    code_original = \
"""\
    def verify(self) -> None:
        if not isinstance(self.data, list):
            raise VerifyException("int_list data should hold a list.")
        for elem in self.data:
            if not isinstance(elem, int):
                raise VerifyException(
                    "int_list list elements should be integers.")
"""
    code_original = code_original.splitlines(True)
    code_formatted =\
"""\
 37    def verify(self) -> None:
 38        if not isinstance(self.data, list):
 39            raise VerifyException("int_list data should hold a list.")
 40        for elem in self.data:
 41            if not isinstance(elem, int):
 42                raise VerifyException(
\x1b[0m\x1b[31m 43                    "int_list list elements should be integers.")
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
\x1b[33m 44

"""
    stream = StringIO()
    print(x.extract_code(gen[0][0], 37, 43, code_original), file=stream)
    y = stream.getvalue()
    assert code_formatted in y
