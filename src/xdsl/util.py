from inspect import signature
from typing import Union, NoReturn, Callable, List, overload
import typing
from dataclasses import dataclass
from xdsl.dialects.builtin import FuncOp, FunctionType, ModuleOp, StringAttr
from xdsl.ir import Operation, SSAValue, BlockArgument, Block, Region, Attribute


def assert_never(value: NoReturn) -> NoReturn:
    assert False, f'Unhandled value: {value} ({type(value).__name__})'


OpOrBlockArg = Union[Operation, BlockArgument]


def get_ssa_value(x: OpOrBlockArg) -> SSAValue:
    if isinstance(x, Operation):
        return x.results[0]
    elif isinstance(x, BlockArgument):
        return x
    else:
        assert_never(x)


def new_op(op_name, num_results, num_operands,
           num_regions) -> typing.Type[Operation]:
    @dataclass(eq=False)
    class OpBase(Operation):
        name: str = op_name

        def verify_(self) -> None:
            if len(self.results) != num_results or len(
                    self.operands) != num_operands or len(
                        self.regions) != num_regions:
                raise Exception("%s verifier" % op_name)

    return OpBase


def new_type(type_name):
    @dataclass(frozen=True, eq=False)
    class TypeBase(Attribute):
        name: str = type_name

        def verify_(self) -> None:
            if len(self.parameters) != 0:
                raise Exception(f"{type_name} should have no parameters")

    return TypeBase


@overload
def block(t: Attribute, f: Callable[[BlockArgument],
                                    List[Operation]]) -> Block:
    ...


@overload
def block(
        ts: List[Attribute], f: Callable[[BlockArgument, BlockArgument],
                                         List[Operation]]) -> Block:
    ...


@overload
def block(
    ts: List[Attribute],
    f: Callable[[BlockArgument, BlockArgument, BlockArgument], List[Operation]]
) -> Block:
    ...


def block(t, f) -> Block:
    if isinstance(t, Attribute) and callable(f) and len(
            signature(f).parameters) == 1:
        b = Block.with_arg_types([t])
        b.add_ops(f(b.args[0]))
        return b
    if isinstance(t, list) and callable(f) and len(
            signature(f).parameters) == len(t):
        b = Block.with_arg_types(t)
        b.add_ops(f(*b.args))
        return b


@overload
def func(name: str, type: Attribute, return_type: Attribute,
         f: Callable[[BlockArgument], List[Operation]]) -> Operation:
    ...


@overload
def func(
        name: str, types: List[Attribute], return_type: Attribute,
        f: Callable[[BlockArgument, BlockArgument],
                    List[Operation]]) -> Operation:
    ...


@overload
def func(
    name: str, types: List[Attribute], return_type: Attribute,
    f: Callable[[BlockArgument, BlockArgument, BlockArgument], List[Operation]]
) -> Operation:
    ...


def func(name, input_types, return_types, f) -> Operation:
    type_attr = FunctionType.get(input_types, return_types)
    op = FuncOp.create(
        [], [],
        attributes={
            "sym_name": StringAttr.get(name),
            "type": type_attr,
            "sym_visibility": StringAttr.get("private")
        },
        regions=[Region([block(input_types, f)])])
    return op


# This function is easier to use while parsing, but makes the
# inline definitions complicated
def func2(name, input_types, return_types, region: Region) -> Operation:
    type_attr = FunctionType.get(input_types, return_types)
    op = FuncOp.create(
        [], [],
        attributes={
            "sym_name": StringAttr.get(name),
            "type": type_attr,
            "sym_visibility": StringAttr.get("private")
        },
        regions=[region])
    return op


def module(ops: Union[List[Operation], Region]) -> Operation:
    if isinstance(ops, List):
        region = Region([Block([], ops)])
    else:
        region = ops
    op = ModuleOp.create([], [], regions=[region])
    return op
