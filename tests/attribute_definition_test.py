"""
Test the definition of attributes and their constraints.
"""

from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from typing import Any, List, Type, TypeGuard, TypeVar, cast

import pytest

from xdsl.ir import Attribute, Data, ParametrizedAttribute
from xdsl.irdl import AllOf, AttrConstraint, BaseAttr, GenericData, ParamAttrConstraint, ParameterDef, VerifyException, attr_constr_coercion, irdl_attr_definition, builder, irdl_to_attr_constraint
from xdsl.parser import Parser
from xdsl.printer import Printer

#  ____        _
# |  _ \  __ _| |_ __ _
# | | | |/ _` | __/ _` |
# | |_| | (_| | || (_| |
# |____/ \__,_|\__\__,_|
#


@irdl_attr_definition
class BoolData(Data[bool]):
    """An attribute holding a boolean value."""
    name = "bool"

    @staticmethod
    def parse_parameter(parser: Parser) -> bool:
        val = parser.parse_optional_ident()
        if val == "True":
            return True
        elif val == "False":
            return False
        else:
            raise Exception("Wrong argument passed to BoolAttr.")

    @staticmethod
    def print_parameter(data: bool, printer: Printer):
        printer.print_string(str(data))


def test_simple_data():
    """Test that the definition of a data with a class parameter."""
    b = BoolData(True)
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(b)
    assert stream.getvalue() == "!bool<True>"


def test_simple_data_verifier_failure():
    """
    Test that the verifier of a data with a class parameter fails when given
    a parameter of the wrong type.
    """
    with pytest.raises(VerifyException) as e:
        BoolData(2)  # type: ignore
    assert e.value.args[0] == ("bool data attribute expected type "
                               "<class 'bool'>, but <class 'int'> given.")


class IntListMissingVerifierData(Data[List[int]]):
    """
    An attribute holding a list of integers.
    The definition should fail, since no verifier is provided, and the Data
    type parameter is not a class.
    """
    name = "missing_verifier_data"

    @staticmethod
    def parse_parameter(parser: Parser) -> List[int]:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: List[int], printer: Printer) -> None:
        raise NotImplementedError()


def test_data_with_non_class_param_missing_verifier_failure():
    """
    Test that a non-class Data parameter requires the definition of a verifier.
    """
    with pytest.raises(Exception) as e:
        irdl_attr_definition(IntListMissingVerifierData)
    assert e.value.args[0] == (
        'In IntListMissingVerifierData definition: '
        'Cannot infer "verify" method. Type parameter of Data is not a class.')


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


def test_non_class_data():
    """Test the definition of a Data with a non-class parameter."""
    attr = IntListData([0, 1, 42])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "!int_list<[0, 1, 42]>"


def test_simple_data_constructor_failure():
    """
    Test that the verifier of a Data with a non-class parameter fails when
    given wrong arguments.
    """
    with pytest.raises(VerifyException) as e:
        IntListData([0, 1, 42, ""])  # type: ignore
    assert e.value.args[0] == "int_list list elements should be integers."


#   ____                      _      ____        _
#  / ___| ___ _ __   ___ _ __(_) ___|  _ \  __ _| |_ __ _
# | |  _ / _ \ '_ \ / _ \ '__| |/ __| | | |/ _` | __/ _` |
# | |_| |  __/ | | |  __/ |  | | (__| |_| | (_| | || (_| |
#  \____|\___|_| |_|\___|_|  |_|\___|____/ \__,_|\__\__,_|
#

_MissingGenericDataData = TypeVar("_MissingGenericDataData")


@irdl_attr_definition
class MissingGenericDataData(Data[_MissingGenericDataData]):
    name = "missing_genericdata"

    @staticmethod
    def parse_parameter(parser: Parser) -> _MissingGenericDataData:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: _MissingGenericDataData,
                        printer: Printer) -> None:
        raise NotImplementedError()

    def verify(self) -> None:
        return


class MissingGenericDataDataWrapper(ParametrizedAttribute):
    name = "missing_genericdata_wrapper"

    param: ParameterDef[MissingGenericDataData[int]]


def test_data_with_generic_missing_generic_data_failure():
    """
    Test error message when a generic data is used in constraints
    without implementing GenericData.
    """
    with pytest.raises(Exception) as e:
        irdl_attr_definition(MissingGenericDataDataWrapper)
    assert e.value.args[0] == (
        "Generic `Data` type 'missing_genericdata' cannot be converted to "
        "an attribute constraint. Consider making it inherit from "
        "`GenericData` instead of `Data`.")


A = TypeVar("A", bound=Attribute)


@dataclass
class DataListAttr(AttrConstraint):
    """
    A constraint that enforces that the elements of a ListData all respect
    a constraint.
    """
    elem_constr: AttrConstraint

    def verify(self, attr: Attribute) -> None:
        attr = cast(ListData, attr)
        for e in attr.data:
            self.elem_constr.verify(e)


@irdl_attr_definition
class ListData(GenericData[List[A]]):
    name = "list"

    @staticmethod
    def parse_parameter(parser: Parser) -> List[A]:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: List[A], printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        assert len(args) == 1
        return DataListAttr(irdl_to_attr_constraint(args[0]))

    @staticmethod
    @builder
    def from_list(data: List[A]) -> ListData[A]:
        return ListData(data)

    def verify(self) -> None:
        if not isinstance(self.data, list):
            raise VerifyException(
                f"Wrong type given to attribute {self.name}: got"
                f" {type(self.data)}, but expected list of"
                " attributes.")
        for idx, val in enumerate(self.data):
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but element "
                    f"{idx} is of type {type(val)}.")


def test_generic_data_verifier():
    """
    Test that a GenericData can be created.
    """
    attr = ListData([BoolData(True), ListData([BoolData(False)])])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue() == "!list<[!bool<True>, !list<[!bool<False>]>]>"


def test_generic_data_verifier_fail():
    """
    Test that a GenericData verifier fails when given wrong parameters.
    """
    with pytest.raises(VerifyException) as e:
        ListData([0])  # type: ignore
    assert e.value.args[0] == ("list data expects attribute list, but"
                               " element 0 is of type <class 'int'>.")


@irdl_attr_definition
class ListDataWrapper(ParametrizedAttribute):
    name = "list_wrapper"

    val: ParameterDef[ListData[BoolData]]


def test_generic_data_wrapper_verifier():
    """
    Test that a GenericData used in constraints pass the verifier when correct.
    """
    attr = ListDataWrapper([ListData([BoolData(True), BoolData(False)])])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue(
    ) == "!list_wrapper<!list<[!bool<True>, !bool<False>]>>"


def test_generic_data_wrapper_verifier_failure():
    """
    Test that a GenericData used in constraints fails
    the verifier when constraints are not satisfied.
    """
    with pytest.raises(VerifyException) as e:
        ListDataWrapper(
            [ListData([BoolData(True),
                       ListData([BoolData(False)])])])
    assert e.value.args[
        0] == "ListData(data=[BoolData(data=False)]) should be of base attribute bool"


@irdl_attr_definition
class ListDataNoGenericsWrapper(ParametrizedAttribute):
    name = "list_no_generics_wrapper"

    val: ParameterDef[ListData]


def test_generic_data_no_generics_wrapper_verifier():
    """
    Test that GenericType can be used in constraints without a parameter.
    """
    attr = ListDataNoGenericsWrapper(
        [ListData([BoolData(True), ListData([BoolData(False)])])])
    stream = StringIO()
    p = Printer(stream=stream)
    p.print_attribute(attr)
    assert stream.getvalue(
    ) == "!list_no_generics_wrapper<!list<[!bool<True>, !list<[!bool<False>]>]>>"
