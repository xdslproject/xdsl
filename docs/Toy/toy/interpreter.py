import operator
from typing import Any
from itertools import accumulate

from dataclasses import dataclass

from xdsl.dialects.builtin import TensorType, VectorType, ModuleOp
from xdsl.interpreter import Interpreter, InterpreterFunctionTable
from xdsl.utils.exceptions import IntepretationError

from . import dialect as toy


def run_toy_func(intepreter: Interpreter, name: str,
                 args: tuple[Any, ...]) -> tuple[Any, ...]:
    for op in intepreter.module.ops:
        if isinstance(op, toy.FuncOp) and op.sym_name.data == name:
            return run_func(intepreter, op, args)

    raise IntepretationError(f'Could not find toy function with name: {name}')


toy_ft = InterpreterFunctionTable()


@dataclass
class Tensor:
    data: list[float]
    shape: list[int]

    def __format__(self, __format_spec: str) -> str:
        prod_shapes: list[int] = list(
            accumulate(reversed(self.shape), operator.mul))
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
def run_print(interpreter: Interpreter, op: toy.PrintOp,
              args: tuple[Any, ...]) -> tuple[Any, ...]:
    interpreter.print(f'{args[0]}')
    return ()


@toy_ft.register(toy.FuncOp)
def run_func(interpreter: Interpreter, op: toy.FuncOp,
             args: tuple[Any, ...]) -> tuple[Any, ...]:
    interpreter.push_scope(f'ctx_{op.sym_name.data}')
    block = op.body.blocks[0]
    interpreter.set_values(block.args, args)
    for body_op in block.ops:
        interpreter.run(body_op)
    assert isinstance(block.ops[-1], toy.ReturnOp)
    results = interpreter.get_values(tuple(block.ops[-1].operands))
    interpreter.pop_scope()
    return results


@toy_ft.register(toy.ConstantOp)
def run_const(interpreter: Interpreter, op: toy.ConstantOp,
              args: tuple[Any, ...]) -> tuple[Any, ...]:
    assert not len(args)
    data = op.get_data()
    shape = op.get_shape()
    result = Tensor(data, shape)
    return result,


@toy_ft.register(toy.ReshapeOp)
def run_reshape(interpreter: Interpreter, op: toy.ReshapeOp,
                args: tuple[Any, ...]) -> tuple[Any, ...]:
    arg, = args
    assert isinstance(arg, Tensor)
    result_typ = op.results[0].typ
    assert isinstance(result_typ, VectorType | TensorType)
    new_shape = result_typ.get_shape()

    return Tensor(arg.data, new_shape),


@toy_ft.register(toy.AddOp)
def run_add(interpreter: Interpreter, op: toy.AddOp,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
    lhs, rhs = args
    assert isinstance(lhs, Tensor)
    assert isinstance(rhs, Tensor)
    assert lhs.shape == rhs.shape

    return Tensor([l + r for l, r in zip(lhs.data, rhs.data)], lhs.shape),


@toy_ft.register(toy.MulOp)
def run_mul(interpreter: Interpreter, op: toy.MulOp,
            args: tuple[Any, ...]) -> tuple[Any, ...]:
    lhs, rhs = args
    assert isinstance(lhs, Tensor)
    assert isinstance(rhs, Tensor)
    assert lhs.shape == rhs.shape

    return Tensor([l * r for l, r in zip(lhs.data, rhs.data)], lhs.shape),


@toy_ft.register(toy.ReturnOp)
def run_return(interpreter: Interpreter, op: toy.ReturnOp,
               args: tuple[Any, ...]) -> tuple[Any, ...]:
    assert len(args) < 2
    return ()


@toy_ft.register(toy.GenericCallOp)
def run_generic_call(interpreter: Interpreter, op: toy.GenericCallOp,
                     args: tuple[Any, ...]) -> tuple[Any, ...]:
    return run_toy_func(interpreter, op.callee.string_value(), args)


@toy_ft.register(toy.TransposeOp)
def run_transpose(interpreter: Interpreter, op: toy.TransposeOp,
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


@toy_ft.register(ModuleOp)
def run_module(interpreter: Interpreter, op: ModuleOp,
               args: tuple[Any, ...]) -> tuple[Any, ...]:
    return run_toy_func(interpreter, 'main', args)
