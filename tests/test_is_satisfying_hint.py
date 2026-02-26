from typing import Any, Generic, Literal, TypeAlias

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatData,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    i32,
)
from xdsl.ir import Attribute, ParametrizedAttribute, SSAValue
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.hints import isa
from xdsl.utils.test_value import create_ssa_value


class Class1:
    pass


class SubClass1(Class1):
    pass


class Class2:
    pass


################################################################################
# Class
################################################################################


def test_class_hint_correct():
    """Test that a class hint is being satisfied by giving a class instance."""
    assert isa(Class1(), Class1)
    assert isa(SubClass1(), SubClass1)
    assert isa(Class2(), Class2)
    assert isa(True, bool)
    assert isa(10, int)
    assert isa("", str)


def test_class_hint_class():
    """Test that a class hint is not being satisfied by the class object."""
    assert not isa(Class1, Class1)


def test_class_hint_wrong_instance():
    """
    Test that a class hint is not being satisfied by another class instance.
    """
    assert not isa(Class2(), Class1)
    assert not isa(0, Class1)
    assert not isa("", Class1)
    assert not isa("", int)


def test_class_hint_subclass():
    """
    Test that a class hint is satisfied by a subclass.
    """
    assert isa(SubClass1(), Class1)
    assert isa(True, int)  # bool is a subclass of int


################################################################################
# List
################################################################################


def test_list_hint_empty():
    """Test that empty lists satisfy all list hints."""
    assert isa([], list[int])
    assert isa([], list[bool])
    assert isa([], list[Class1])


def test_list_hint_correct():
    """
    Test that list hints work correcly on non-empty lists of the right type.
    """
    assert isa([42], list[int])
    assert isa([0, 3, 5], list[int])
    assert isa([False], list[bool])
    assert isa([True, False], list[bool])
    assert isa([Class1(), SubClass1()], list[Class1])


def test_list_hint_not_list_failure():
    """Test that list hints work correcly on non lists."""
    assert not isa(0, list[int])
    assert not isa(0, list[Any])
    assert not isa(True, list[bool])
    assert not isa(True, list[Any])
    assert not isa("", list[Any])
    assert not isa("", list[str])
    assert not isa(Class1(), list[Class1])
    assert not isa(Class1(), list[Any])
    assert not isa({}, list[dict[Any, Any]])
    assert not isa({}, list[Any])


def test_list_hint_failure():
    """
    Test that list hints work correcly on non-empty lists of the wrong type.
    """
    assert not isa([0], list[bool])
    assert not isa([0, True], list[bool])
    assert not isa([True, 0], list[bool])
    assert not isa([True, False, True, 0], list[bool])
    assert not isa([Class2()], list[Class1])


def test_list_hint_nested():
    """
    Test that we can check nested list hints.
    """
    assert isa([[]], list[list[int]])
    assert isa([[0]], list[list[int]])
    assert isa([[0], [2, 1]], list[list[int]])
    assert not isa([[0], [2, 1], [""]], list[list[int]])
    assert not isa([[0], [2, 1], [32, ""]], list[list[int]])
    assert isa([], list[list[list[int]]])
    assert isa([[]], list[list[list[int]]])
    assert isa([[[]]], list[list[list[int]]])
    assert isa([[], [[]]], list[list[list[int]]])
    assert isa([[], [[0, 32]]], list[list[list[int]]])


################################################################################
# Tuple
################################################################################


def test_tuple_hint_empty():
    """Test that empty tuple satisfy all tuple hints."""
    assert isa(tuple(), tuple[int, ...])
    assert isa(tuple(), tuple[bool, ...])
    assert isa(tuple(), tuple[Class1, ...])


def test_tuple_hint_correct():
    """
    Test that tuple hints work correcly on non-empty tuples of the right type.
    """
    assert isa((42,), tuple[int])
    assert isa((0, 3, 5), tuple[int, int, int])
    assert isa((False,), tuple[bool])
    assert isa((True, False), tuple[bool, ...])
    assert isa((True, 1, "test"), tuple[bool, int, str])
    assert isa((Class1(), SubClass1()), tuple[Class1, ...])


def test_tuple_hint_not_list_failure():
    """Test that tuple hints work correcly on non tuple."""
    assert not isa(0, tuple[int])
    assert not isa(0, tuple[Any])
    assert not isa(True, tuple[bool])
    assert not isa(True, tuple[Any])
    assert not isa("", tuple[Any])
    assert not isa("", tuple[str])
    assert not isa(Class1(), tuple[Class1])
    assert not isa(Class1(), tuple[Any])
    assert not isa({}, tuple[dict[Any, Any]])
    assert not isa({}, tuple[Any])


def test_tuple_hint_failure():
    """
    Test that tuple hints work correcly on non-empty tuples of the wrong type.
    """
    assert not isa((0,), tuple[bool])
    assert not isa((0, True), tuple[bool, bool])
    assert not isa((0, True), tuple[int])
    assert not isa((True, 0), tuple[bool, bool])
    assert not isa((True, False, True, 0), tuple[bool, ...])
    assert not isa((Class2(),), tuple[Class1])


def test_tuple_hint_nested():
    """
    Test that we can check nested tuple hints.
    """
    assert isa(((),), tuple[tuple[int, ...]])
    assert isa(((0,),), tuple[tuple[int]])
    assert isa((0, (1, 2)), tuple[int, tuple[int, int]])
    assert isa(((0, 1), (2, 3), (4, 5)), tuple[tuple[int, int], ...])
    assert not isa(((0, 1), (2, 3), (4, "5")), tuple[tuple[int, int], ...])
    assert not isa(((0, 1), (2, 3), (4, "5")), tuple[tuple[int, ...], ...])


################################################################################
# Set
################################################################################


def test_set_hint_empty():
    """Test that empty set satisfy all set hints."""
    assert isa(set(), set[int])
    assert isa(set(), set[bool])
    assert isa(set(), set[Class1])


def test_set_hint_correct():
    """
    Test that set hints work correcly on non-empty sets of the right type.
    """
    assert isa({42}, set[int])
    assert isa({0, 3, 5}, set[int])
    assert isa({False}, set[bool])
    assert isa({True, False}, set[bool])
    assert isa({True, 1, "test"}, set[bool | int | str])
    assert isa({Class1(), SubClass1()}, set[Class1])


def test_set_hint_not_list_failure():
    """Test that set hints work correcly on non set."""
    assert not isa(0, set[int])
    assert not isa(0, set[Any])
    assert not isa(True, set[bool])
    assert not isa(True, set[Any])
    assert not isa("", set[Any])
    assert not isa("", set[str])
    assert not isa(Class1(), set[Class1])
    assert not isa(Class1(), set[Any])
    assert not isa([], set[dict[Any, Any]])
    assert not isa([], set[Any])


def test_set_hint_failure():
    """
    Test that set hints work correcly on non-empty sets of the wrong type.
    """
    assert not isa({0}, set[bool])
    assert not isa({0, "hello"}, set[int])


################################################################################
# Dictionary
################################################################################


def test_dict_hint_empty():
    """Test that empty dicts satisfy all dict hints."""
    assert isa({}, dict[int, bool])
    assert isa({}, dict[Any, Any])
    assert isa({}, dict[Class1, Any])
    assert isa({}, dict[str, int])


def test_dict_hint_correct():
    """
    Test that dict hints work correcly on non-empty dicts of the right type.
    """
    assert isa({"": 0}, dict[str, int])
    assert isa({"": 0, "a": 32}, dict[str, int])
    assert isa({0: "", 32: "a"}, dict[int, str])
    assert isa({True: Class1(), False: SubClass1()}, dict[bool, Class1])


def test_dict_hint_not_dict_failure():
    """Test that dict hints work correcly on non dicts."""
    assert not isa(0, dict[int, int])
    assert not isa(0, dict[Any, Any])
    assert not isa(True, dict[bool, bool])
    assert not isa(True, dict[Any, Any])
    assert not isa("", dict[Any, Any])
    assert not isa(Class1(), dict[Any, Any])
    assert not isa([], dict[Any, Any])


def test_dict_hint_failure():
    """
    Test that dict hints work correcly on non-empty dicts of the wrong type.
    """
    assert not isa({"": ""}, dict[int, str])
    assert not isa({0: 0}, dict[int, str])
    assert not isa({0: "0", 2: 2}, dict[int, str])
    assert not isa({0: "0", 2: "1", "2": "2"}, dict[int, str])


ThreeDict: TypeAlias = dict[int, dict[int, dict[int, str]]]


def test_dict_hint_nested():
    """
    Test that we can check nested dict hints.
    """
    assert isa({}, dict[int, dict[int, str]])
    assert isa({0: {}}, dict[int, dict[int, str]])
    assert isa({0: {1: ""}}, dict[int, dict[int, str]])

    assert not isa({"": {}}, dict[int, dict[int, str]])
    assert not isa({0: ""}, dict[int, dict[int, str]])
    assert not isa({0: {"": ""}}, dict[int, dict[int, str]])
    assert not isa({0: {0: 0}}, dict[int, dict[int, str]])

    assert isa({}, ThreeDict)
    assert isa({0: {}}, ThreeDict)
    assert isa({0: {0: {}}}, ThreeDict)
    assert isa({0: {}, 1: {0: {}}}, ThreeDict)
    assert isa({0: {}, 1: {0: {0: "0", 1: "32"}}}, ThreeDict)


################################################################################
# GenericData
################################################################################


def test_generic_data():
    attr = ArrayAttr([IntAttr(0)])

    assert isa(attr, ArrayAttr[IntAttr])
    assert isa(attr, ArrayAttr[IntAttr | FloatData])
    assert not isa(attr, ArrayAttr[FloatData])

    attr2 = ArrayAttr([IntAttr(0), FloatData(0.0)])

    assert not isa(attr2, ArrayAttr[IntAttr])
    assert isa(attr2, ArrayAttr[IntAttr | FloatData])
    assert not isa(attr2, ArrayAttr[FloatData])

    intattr = IntAttr(42)

    assert not isa(intattr, ArrayAttr[Attribute])
    assert not isa(intattr, DictionaryAttr)

    integerattr = IntegerAttr.from_index_int_value(42)

    assert not isa(integerattr, ArrayAttr[Attribute])
    assert not isa(integerattr, DictionaryAttr)


def test_nested_generic_data():
    attr = ArrayAttr([IntegerAttr.from_index_int_value(0)])

    assert isa(attr, ArrayAttr[Attribute])
    assert isa(attr, ArrayAttr[IntegerAttr[IndexType]])
    assert isa(attr, ArrayAttr[IntegerAttr[IndexType | IntegerType]])
    assert not isa(attr, ArrayAttr[IntegerAttr[IntegerType]])


################################################################################
# ParametrizedAttribute
################################################################################

_T = TypeVar("_T", bound=Attribute)


@irdl_attr_definition
class MyParamAttr(ParametrizedAttribute, Generic[_T]):
    name = "test.param"

    v: _T


def test_parametrized_attribute():
    attr = MyParamAttr[IntAttr](IntAttr(0))

    # `assert isa(attr, MyParamAttr)` not supported: use isinstance instead
    assert isa(attr, MyParamAttr[IntAttr])
    assert isa(attr, MyParamAttr[IntAttr | FloatData])
    assert not isa(attr, MyParamAttr[FloatData])


################################################################################
# Literal
################################################################################


def test_literal():
    assert isa("string", Literal["string"])
    assert isa("string", Literal["string", "another string"])
    assert isa("another string", Literal["string", "another string"])
    assert not isa("not this string", Literal["string", "another string"])

    assert isa(1, Literal[1])
    assert isa(1, Literal[1, 2])
    assert isa(2, Literal[1, 2])
    assert not isa(3, Literal[1, 2])

    assert not isa(1, Literal["1"])
    assert not isa("1", Literal[1])


################################################################################
# SSAValue
################################################################################


def test_ssavalue():
    a = create_ssa_value(i32)

    assert isa(a, SSAValue)
    assert isa(a, SSAValue[IntegerType])
    assert not isa(a, SSAValue[StringAttr])
    assert not isa(a, SSAValue[IntegerAttr[IntegerType]])

    b = create_ssa_value(IntegerAttr(2, i32))

    assert isa(b, SSAValue[IntegerAttr[IntegerType]])
    assert not isa(b, SSAValue[IntegerAttr[IndexType]])
    assert not isa(b, SSAValue[IntegerType])
