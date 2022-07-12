from xdsl.util import is_satisfying_hint
from typing import Any, TypeAlias


class Class1:
    pass


class SubClass1(Class1):
    pass


class Class2:
    pass


#     _
#    / \   _ __  _   _
#   / _ \ | '_ \| | | |
#  / ___ \| | | | |_| |
# /_/   \_\_| |_|\__, |
#                |___/
#


def test_any_hint():
    """Test that the we can check if a value is satisfying `Any`."""
    assert is_satisfying_hint(3, Any)
    assert is_satisfying_hint([], Any)
    assert is_satisfying_hint([2], Any)
    assert is_satisfying_hint(int, Any)


#   ____ _
#  / ___| | __ _ ___ ___
# | |   | |/ _` / __/ __|
# | |___| | (_| \__ \__ \
#  \____|_|\__,_|___/___/
#


def test_class_hint_correct():
    """Test that a class hint is being satisfied by giving a class instance."""
    assert is_satisfying_hint(Class1(), Class1)
    assert is_satisfying_hint(SubClass1(), SubClass1)
    assert is_satisfying_hint(Class2(), Class2)
    assert is_satisfying_hint(True, bool)
    assert is_satisfying_hint(10, int)
    assert is_satisfying_hint("", str)


def test_class_hint_class():
    """Test that a class hint is not being satisfied by the class object."""
    assert not is_satisfying_hint(Class1, Class1)


def test_class_hint_wrong_instance():
    """
    Test that a class hint is not being satisfied by another class instance.
    """
    assert not is_satisfying_hint(Class2(), Class1)
    assert not is_satisfying_hint(0, Class1)
    assert not is_satisfying_hint("", Class1)
    assert not is_satisfying_hint("", int)


def test_class_hint_subclass():
    """
    Test that a class hint is not being satisfied by a non-class object.
    """
    assert is_satisfying_hint(SubClass1(), Class1)
    assert is_satisfying_hint(True, int)  # bool is a subclass of int


#  _     _     _
# | |   (_)___| |_
# | |   | / __| __|
# | |___| \__ \ |_
# |_____|_|___/\__|
#


def test_list_hint_empty():
    """Test that empty lists satisfy all list hints."""
    assert is_satisfying_hint([], list[int])
    assert is_satisfying_hint([], list[bool])
    assert is_satisfying_hint([], list[Class1])


def test_list_hint_correct():
    """
    Test that list hints work correcly on non-empty lists of the right type.
    """
    assert is_satisfying_hint([42], list[int])
    assert is_satisfying_hint([0, 3, 5], list[int])
    assert is_satisfying_hint([False], list[bool])
    assert is_satisfying_hint([True, False], list[bool])
    assert is_satisfying_hint([Class1(), SubClass1()], list[Class1])


def test_list_hint_not_list_failure():
    """Test that list hints work correcly on non lists."""
    assert not is_satisfying_hint(0, list[int])
    assert not is_satisfying_hint(0, list[Any])
    assert not is_satisfying_hint(True, list[bool])
    assert not is_satisfying_hint(True, list[Any])
    assert not is_satisfying_hint("", list[Any])
    assert not is_satisfying_hint("", list[str])
    assert not is_satisfying_hint(Class1(), list[Class1])
    assert not is_satisfying_hint(Class1(), list[Any])
    assert not is_satisfying_hint({}, list[dict[Any, Any]])
    assert not is_satisfying_hint({}, list[Any])


def test_list_hint_failure():
    """
    Test that list hints work correcly on non-empty lists of the wrong type.
    """
    assert not is_satisfying_hint([0], list[bool])
    assert not is_satisfying_hint([0, True], list[bool])
    assert not is_satisfying_hint([True, 0], list[bool])
    assert not is_satisfying_hint([True, False, True, 0], list[bool])
    assert not is_satisfying_hint([Class2()], list[Class1])


def test_list_hint_nested():
    """
    Test that we can check nested list hints.
    """
    assert is_satisfying_hint([[]], list[list[int]])
    assert is_satisfying_hint([[0]], list[list[int]])
    assert is_satisfying_hint([[0], [2, 1]], list[list[int]])
    assert not is_satisfying_hint([[0], [2, 1], [""]], list[list[int]])
    assert not is_satisfying_hint([[0], [2, 1], [32, ""]], list[list[int]])
    assert is_satisfying_hint([], list[list[list[int]]])
    assert is_satisfying_hint([[]], list[list[list[int]]])
    assert is_satisfying_hint([[[]]], list[list[list[int]]])
    assert is_satisfying_hint([[], [[]]], list[list[list[int]]])
    assert is_satisfying_hint([[], [[0, 32]]], list[list[list[int]]])


#   ____  _      _
# |  _ \(_) ___| |_
# | | | | |/ __| __|
# | |_| | | (__| |_
# |____/|_|\___|\__|
#


def test_dict_hint_empty():
    """Test that empty dicts satisfy all dict hints."""
    assert is_satisfying_hint({}, dict[int, bool])
    assert is_satisfying_hint({}, dict[Any, Any])
    assert is_satisfying_hint({}, dict[Class1, Any])
    assert is_satisfying_hint({}, dict[str, int])


def test_dict_hint_correct():
    """
    Test that dict hints work correcly on non-empty dicts of the right type.
    """
    assert is_satisfying_hint({"": 0}, dict[str, int])
    assert is_satisfying_hint({"": 0, "a": 32}, dict[str, int])
    assert is_satisfying_hint({0: "", 32: "a"}, dict[int, str])
    assert is_satisfying_hint({
        True: Class1(),
        False: SubClass1()
    }, dict[bool, Class1])


def test_dict_hint_not_dict_failure():
    """Test that dict hints work correcly on non dicts."""
    assert not is_satisfying_hint(0, dict[int, int])
    assert not is_satisfying_hint(0, dict[Any, Any])
    assert not is_satisfying_hint(True, dict[bool, bool])
    assert not is_satisfying_hint(True, dict[Any, Any])
    assert not is_satisfying_hint("", dict[Any, Any])
    assert not is_satisfying_hint(Class1(), dict[Any, Any])
    assert not is_satisfying_hint([], dict[Any, Any])


def test_dict_hint_failure():
    """
    Test that dict hints work correcly on non-empty dicts of the wrong type.
    """
    assert not is_satisfying_hint({"": ""}, dict[int, str])
    assert not is_satisfying_hint({0: 0}, dict[int, str])
    assert not is_satisfying_hint({0: "0", 2: "1", 2: 2}, dict[int, str])
    assert not is_satisfying_hint({0: "0", 2: "1", "2": "2"}, dict[int, str])


threeDict: TypeAlias = dict[int, dict[int, dict[int, str]]]


def test_dict_hint_nested():
    """
    Test that we can check nested dict hints.
    """
    assert is_satisfying_hint({}, dict[int, dict[int, str]])
    assert is_satisfying_hint({0: {}}, dict[int, dict[int, str]])
    assert is_satisfying_hint({0: {1: ""}}, dict[int, dict[int, str]])

    assert not is_satisfying_hint({"": {}}, dict[int, dict[int, str]])
    assert not is_satisfying_hint({0: ""}, dict[int, dict[int, str]])
    assert not is_satisfying_hint({0: {"": ""}}, dict[int, dict[int, str]])
    assert not is_satisfying_hint({0: {0: 0}}, dict[int, dict[int, str]])

    assert is_satisfying_hint({}, threeDict)
    assert is_satisfying_hint({0: {}}, threeDict)
    assert is_satisfying_hint({0: {0: {}}}, threeDict)
    assert is_satisfying_hint({0: {}, 1: {0: {}}}, threeDict)
    assert is_satisfying_hint({0: {}, 1: {0: {0: 0, 1: 32}}}, threeDict)
