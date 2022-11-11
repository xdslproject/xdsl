from __future__ import annotations

import pytest

from xdsl.ir import ParametrizedAttribute, Data
from xdsl.irdl import irdl_attr_definition, builder
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class NoBuilderAttr(ParametrizedAttribute):
    name = "test.no_builder_attr"


def test_no_builder_default():
    attr = NoBuilderAttr.build(NoBuilderAttr())
    assert attr == NoBuilderAttr()


def test_no_builder_exception():
    with pytest.raises(TypeError):
        NoBuilderAttr.build(3)


@irdl_attr_definition
class OneBuilderAttr(Data[str]):
    name = "test.one_builder_attr"

    @staticmethod
    @builder
    def from_int(data: int) -> OneBuilderAttr:
        return OneBuilderAttr(str(data))

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        pass

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        pass


def test_one_builder_default():
    attr = OneBuilderAttr.build(OneBuilderAttr("a"))
    assert attr == OneBuilderAttr("a")


def test_one_builder_builder():
    attr = OneBuilderAttr.build(1)
    assert attr == OneBuilderAttr("1")


def test_one_builder_exception():
    with pytest.raises(TypeError):
        OneBuilderAttr.build("1")


@irdl_attr_definition
class TwoBuildersAttr(Data[str]):
    name = "test.two_builder_attr"

    @staticmethod
    @builder
    def from_int(data: int) -> TwoBuildersAttr:
        return TwoBuildersAttr(str(data))

    @staticmethod
    @builder
    def from_str(s: str) -> TwoBuildersAttr:
        return TwoBuildersAttr(s)

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        pass

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        pass


def test_two_builders_default():
    attr = TwoBuildersAttr.build(TwoBuildersAttr("a"))
    assert attr == TwoBuildersAttr("a")


def test_two_builders_first_builder():
    attr = TwoBuildersAttr.build(1)
    assert attr == TwoBuildersAttr("1")


def test_two_builders_second_builder():
    attr = TwoBuildersAttr.build("1")
    assert attr == TwoBuildersAttr("1")


def test_two_builders_bad_args():
    with pytest.raises(TypeError) as e:
        TwoBuildersAttr.build([])


@irdl_attr_definition
class BuilderDefaultArgAttr(Data[str]):
    name = "test.builder_default_arg_attr"
    param: str

    @staticmethod
    @builder
    def from_int(data: int) -> BuilderDefaultArgAttr:
        return BuilderDefaultArgAttr(str(data))

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        pass

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        pass


def builder_default_arg_default():
    attr = BuilderDefaultArgAttr.build(4)
    assert attr == BuilderDefaultArgAttr("40")


def builder_default_arg_arg():
    attr = BuilderDefaultArgAttr.build(4, 2)
    assert attr == BuilderDefaultArgAttr("42")


@irdl_attr_definition
class BuilderUnionArgAttr(Data[str]):
    name = "test.builder_union_arg_attr"
    param: str

    @staticmethod
    @builder
    def from_int(data: str | int) -> BuilderUnionArgAttr:
        return BuilderUnionArgAttr(str(data))

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        pass

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        pass


def builder_union_arg_first():
    attr = BuilderUnionArgAttr.build(4)
    assert attr == BuilderUnionArgAttr("4")


def builder_union_arg_second():
    attr = BuilderUnionArgAttr.build("4")
    assert attr == BuilderUnionArgAttr("4")


def builder_union_arg_bad_argument():
    with pytest.raises(TypeError) as e:
        BuilderUnionArgAttr.build([])
