from io import StringIO
import operator
from typing import Any
from itertools import accumulate

from dataclasses import dataclass

from xdsl.dialects.builtin import TensorType, VectorType, ModuleOp
from xdsl.interpreter import Emulator, EmulationError, FunctionTable

from . import dialect as toy


def run_toy_func(emulator: Emulator, name: str,
                 args: tuple[Any, ...]) -> tuple[Any, ...]:
    for op in emulator.module.regions[0].blocks[0].ops:
        if isinstance(op, toy.FuncOp) and op.sym_name.data == name:
            return run_func(emulator, op, args)

    raise EmulationError(f'Could not find toy function with name: {name}')


toy_ft = FunctionTable()


@dataclass
class Tensor:
    data: list[float]
    shape: list[int]

    def __format__(self, __format_spec: str) -> str:
        prod_shapes: list[int] = list(
            accumulate(self.shape[::-1], operator.mul))
        assert prod_shapes[-1] == len(self.data)
        result = '[' * len(self.shape)

        for i, d in enumerate(self.data):
            if i:
                n = sum(not i % p for p in prod_shapes)
                result += ']' * n
                result += ', '
                result += '[' * n
            result += f'{d}'

        result += ']' * len(self.shape)
        return result


@toy_ft.register(toy.PrintOp)
def run_print(emulator: Emulator, op: toy.PrintOp,
              args: tuple[Any, ...]) -> tuple[Any, ...]:
    emulator.print(f'{args[0]}')
    return ()


@toy_ft.register(toy.FuncOp)
def run_func(emulator: Emulator, op: toy.FuncOp,
             args: tuple[Any, ...]) -> tuple[Any, ...]:
    emulator.push_context(f'ctx_{op.sym_name.data}')
    block = op.body.blocks[0]
    emulator.set_values(block.args, args)
    for body_op in block.ops:
        emulator.run(body_op)
    assert isinstance(block.ops[-1], toy.ReturnOp)
    results = emulator.get_values(tuple(block.ops[-1].operands))
    emulator.pop_context()
    return results


@toy_ft.register(toy.ConstantOp)
def run_const(emulator: Emulator, op: toy.ConstantOp,
              args: tuple[Any, ...]) -> tuple[Any, ...]:
    assert not len(args)
    data = op.get_data()
    shape = op.get_shape()
    result = Tensor(data, shape)
    return result,


@toy_ft.register(toy.ReshapeOp)
def run_reshape(emulator: Emulator, op: toy.ReshapeOp,
                args: tuple[Any, ...]) -> tuple[Any, ...]:
    arg, = args
    assert isinstance(arg, Tensor)
    result_typ = op.results[0].typ
    assert isinstance(result_typ, VectorType | TensorType)
    new_shape = result_typ.get_shape()

    return Tensor(arg.data, new_shape),


@toy_ft.register(toy.AddOp)
def run_add(emulator: Emulator, op: toy.AddOp,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
    lhs, rhs = args
    assert isinstance(lhs, Tensor)
    assert isinstance(rhs, Tensor)
    assert lhs.shape == rhs.shape

    return Tensor([l + r for l, r in zip(lhs.data, rhs.data)], lhs.shape),


@toy_ft.register(toy.MulOp)
def run_mul(emulator: Emulator, op: toy.MulOp,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
    lhs, rhs = args
    assert isinstance(lhs, Tensor)
    assert isinstance(rhs, Tensor)
    assert lhs.shape == rhs.shape

    return Tensor([l * r for l, r in zip(lhs.data, rhs.data)], lhs.shape),


@toy_ft.register(toy.ReturnOp)
def run_return(emulator: Emulator, op: toy.ReturnOp,
               args: tuple[Any, ...]) -> tuple[Any, ...]:
    assert len(args) < 2
    return ()


@toy_ft.register(toy.GenericCallOp)
def run_generic_call(emulator: Emulator, op: toy.GenericCallOp,
                     args: tuple[Any, ...]) -> tuple[Any, ...]:
    return run_toy_func(emulator, op.callee.string_value(), args)


@toy_ft.register(toy.TransposeOp)
def run_transpose(emulator: Emulator, op: toy.TransposeOp,
                  args: tuple[Any, ...]) -> tuple[Any, ...]:
    arg, = args
    assert isinstance(arg, Tensor)
    assert len(arg.shape) == 2

    cols = arg.shape[0]
    rows = arg.shape[1]

    new_data = [
        arg.data[row * cols + col] for col in range(cols)
        for row in range(rows)
    ]

    result = Tensor(new_data, arg.shape[::-1])

    return result,


def emulate_toy(module: ModuleOp, file: StringIO | None = None):
    emulator = Emulator(module, file=file)
    emulator.register_functions(toy_ft)
    run_toy_func(emulator, 'main', ())
