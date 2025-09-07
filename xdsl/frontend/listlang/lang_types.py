from collections.abc import Sequence
from dataclasses import dataclass

import xdsl.frontend.listlang.list_dialect as list_dialect
from xdsl.builder import Builder
from xdsl.dialects import builtin, printf
from xdsl.frontend.listlang.source import Located, ParseError
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.utils.hints import isa


class ListLangType:
    @staticmethod
    def from_xdsl(xdsl_type: Attribute) -> "ListLangType":
        if isa(xdsl_type, builtin.IntegerType[32]):
            return ListLangInt()
        if isa(xdsl_type, builtin.IntegerType[1]):
            return ListLangBool()
        if isa(xdsl_type, builtin.TensorType[builtin.IntegerType[32]]):
            return ListLangList(ListLangInt())
        if isa(xdsl_type, builtin.TensorType[builtin.IntegerType[1]]):
            return ListLangList(ListLangBool())
        raise ValueError("unknown type")

    def __str__(self) -> str: ...
    def xdsl(self) -> Attribute: ...
    def print(self, builder: Builder, value: SSAValue): ...
    def get_method(self, method: str) -> "Method | None":
        return None


@dataclass
class ListLangInt(ListLangType):
    def __str__(self) -> str:
        return "int"

    def xdsl(self) -> builtin.IntegerType:
        return builtin.IntegerType(32)

    def print(self, builder: Builder, value: SSAValue):
        builder.insert_op(printf.PrintFormatOp("{}", value))


@dataclass
class ListLangBool(ListLangType):
    def __str__(self) -> str:
        return "bool"

    def xdsl(self) -> builtin.IntegerType:
        return builtin.IntegerType(1)

    def print(self, builder: Builder, value: SSAValue):
        builder.insert_op(printf.PrintFormatOp("{}", value))


LIST_ELEMENT_TYPE = ListLangBool | ListLangInt


@dataclass
class ListLangList(ListLangType):
    element_type: LIST_ELEMENT_TYPE

    def __str__(self) -> str:
        return f"list<{self.element_type}>"

    def xdsl(self) -> list_dialect.ListType:
        return list_dialect.ListType(self.element_type.xdsl())

    def print(self, builder: Builder, value: SSAValue):
        builder.insert_op(list_dialect.PrintOp(value))

    def get_method(self, method: str) -> "Method | None":
        match method:
            case "len":
                return ListLenMethod()
            case "map":
                return ListMapMethod()
            case _:
                return None


@dataclass
class TypedExpression:
    value: SSAValue
    typ: ListLangType


## Methods


class Method:
    name: str

    def get_lambda_arg_type(self, x: ListLangType) -> Sequence[ListLangType] | None:
        """
        From the type on which the method was invoked, returns the types of the
        arguments of the method's lambda if there is one, or None if there is
        no lambda.
        """
        ...

    def build(
        self,
        builder: Builder,
        x: Located[TypedExpression],
        lambd: Located[tuple[Block, ListLangType]] | None,
    ) -> TypedExpression:
        """
        Builds the method's execution.

        `lambd` contains a free-standing block containing the lambda
        instructions that must be inlined as needed, and the type of the final
        expression of the block. The associated location is the location of
        the result expression.
        """
        ...


class ListLenMethod(Method):
    name = "len"

    def get_lambda_arg_type(self, x: ListLangType) -> Sequence[ListLangType] | None:
        return None

    def build(
        self,
        builder: Builder,
        x: Located[TypedExpression],
        lambd: Located[tuple[Block, ListLangType]] | None,
    ) -> TypedExpression:
        assert lambd is None
        assert isinstance(x.value.typ, ListLangList)

        len_op = builder.insert_op(list_dialect.LengthOp(x.value.value))
        return TypedExpression(len_op.result, ListLangInt())


class ListMapMethod(Method):
    name = "map"

    def get_lambda_arg_type(self, x: ListLangType) -> Sequence[ListLangType] | None:
        assert isinstance(x, ListLangList)
        return (x.element_type,)

    def build(
        self,
        builder: Builder,
        x: Located[TypedExpression],
        lambd: Located[tuple[Block, ListLangType]] | None,
    ) -> TypedExpression:
        assert lambd is not None
        assert isinstance(x.value.typ, ListLangList)

        if not isinstance(lambd.value[1], LIST_ELEMENT_TYPE):
            raise ParseError(lambd.loc.pos, "expected expression of list-element type")

        map_op = builder.insert_op(
            list_dialect.MapOp(
                x.value.value, Region([lambd.value[0]]), lambd.value[1].xdsl()
            )
        )

        return TypedExpression(map_op.result, ListLangList(lambd.value[1]))
