from __future__ import annotations

import pytest

from xdsl.ir import ParametrizedAttribute, Data
from xdsl.irdl import irdl_attr_definition, builder
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import BuilderNotFoundException


@irdl_attr_definition
class NoBuilderAttr(ParametrizedAttribute):
    name = "test.no_builder_attr"


def test_no_builder_default():
    attr = NoBuilderAttr.build(NoBuilderAttr())
    assert attr == NoBuilderAttr()


def test_no_builder_exception():
    with pytest.raises(BuilderNotFoundException):
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
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        raise NotImplementedError()


def test_one_builder_default():
    attr = OneBuilderAttr.build(OneBuilderAttr("a"))
    assert attr == OneBuilderAttr("a")


def test_one_builder_builder():
    attr = OneBuilderAttr.build(1)
    assert attr == OneBuilderAttr("1")


def test_one_builder_exception():
    with pytest.raises(BuilderNotFoundException):
        OneBuilderAttr.build("1")


def test_one_builder_extra_params():
    with pytest.raises(BuilderNotFoundException):
        OneBuilderAttr.build(1, 2)


@irdl_attr_definition
class OneBuilderTwoArgsAttr(Data[str]):
    name = "test.one_builder_attr"

    @staticmethod
    @builder
    def from_int(data1: int, data2: str) -> OneBuilderAttr:
        return OneBuilderAttr(str(data1) + data2)

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        raise NotImplementedError()


def test_one_builder_two_args_not_enough_args():
    with pytest.raises(BuilderNotFoundException):
        OneBuilderTwoArgsAttr.build(1)


def test_one_builder_two_args_extra_args():
    with pytest.raises(BuilderNotFoundException):
        OneBuilderTwoArgsAttr.build(1, "a", "b")


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
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        raise NotImplementedError()


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
    with pytest.raises(BuilderNotFoundException):
        TwoBuildersAttr.build([])


@irdl_attr_definition
class BuilderDefaultArgAttr(Data[str]):
    name = "test.builder_default_arg_attr"

    @staticmethod
    @builder
    def from_int(data1: int,
                 data2: int = 42,
                 data3: int = 17) -> BuilderDefaultArgAttr:
        return BuilderDefaultArgAttr(f"{data1}, {data2}, {data3}")

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        raise NotImplementedError()


def test_builder_default_args_missing_args():
    with pytest.raises(BuilderNotFoundException):
        BuilderDefaultArgAttr.build()


def test_builder_default_args_only_required_args():
    attr = BuilderDefaultArgAttr.build(4)
    assert attr == BuilderDefaultArgAttr("4, 42, 17")


def test_builder_default_args_only_required_args_and_one():
    attr = BuilderDefaultArgAttr.build(4, 5)
    assert attr == BuilderDefaultArgAttr("4, 5, 17")


def test_builder_default_args_all_args():
    attr = BuilderDefaultArgAttr.build(4, 5, 6)
    assert attr == BuilderDefaultArgAttr("4, 5, 6")


def test_builder_default_args_extra_args():
    with pytest.raises(BuilderNotFoundException):
        BuilderDefaultArgAttr.build(4, 5, 6, 7)


@irdl_attr_definition
class BuilderUnionArgAttr(Data[str]):
    name = "test.builder_union_arg_attr"

    @staticmethod
    @builder
    def from_int(data: str | int) -> BuilderUnionArgAttr:
        return BuilderUnionArgAttr(str(data))

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        raise NotImplementedError()

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        raise NotImplementedError()


def test_builder_union_arg_first():
    attr = BuilderUnionArgAttr.build(4)
    assert attr == BuilderUnionArgAttr("4")


def test_builder_union_arg_second():
    attr = BuilderUnionArgAttr.build("4")
    assert attr == BuilderUnionArgAttr("4")


def test_builder_union_arg_bad_argument():
    with pytest.raises(BuilderNotFoundException):
        BuilderUnionArgAttr.build([])
