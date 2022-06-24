from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from typing import Any, List, TypeVar, cast, Annotated, Generic, TypeAlias

import pytest
import sys
from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import (AttrConstraint, GenericData, ParameterDef,
                       VerifyException, irdl_attr_definition, builder,
                       irdl_to_attr_constraint)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.error_format import Frame
import os

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

        dir_path = os.path.dirname(os.path.realpath(__file__))
        x = Frame(e, 1)
        gen = list(x.get_frame(1))
        code = gen[0][-1]
        assert code == x.extract_code(gen[0][0], gen[0][1])
        assert "raise VerifyException" in code
        assert x.extract_code(dir_path + "/format_test.py", 42) == code
