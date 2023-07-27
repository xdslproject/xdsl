from dataclasses import dataclass, field
from typing import Any

import pytest

from xdsl.dialects import builtin
from xdsl.dialects.builtin import IndexType, IntegerType, ModuleOp, i32
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl_cast,
    register_impls,
)
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import TestSSAValue


def test_import_functions():
    @dataclass
    class A(InterpreterFunctions):
        pass

    @register_impls
    @dataclass
    class B(InterpreterFunctions):
        pass

    i = Interpreter(ModuleOp([]))

    i.register_implementations(B())

    with pytest.raises(ValueError) as e:
        i.register_implementations(A())

    assert e.value.args[0] == "Use `@register_impls` on class A"


def test_cast():
    @dataclass
    class Integer:
        value: int = field()

    @dataclass
    class Index:
        value: int

    @register_impls
    @dataclass
    class CastImpls(InterpreterFunctions):
        @impl_cast(IntegerType, IndexType)
        def cast_integer_to_index(
            self,
            input_type: IntegerType,
            output_type: IndexType,
            value: Any,
        ) -> Any:
            assert isinstance(value, Integer)
            return Index(value.value)

    # Dummy module
    module = ModuleOp([])

    interpreter = Interpreter(module)

    interpreter.register_implementations(CastImpls())

    integer = Integer(42)
    index0 = interpreter.cast_value(i32, IndexType(), integer)
    assert isinstance(index0, Index)

    with pytest.raises(
        InterpretationError,
        match="Could not find cast implementation for types index, i32",
    ):
        integer = interpreter.cast_value(IndexType(), i32, index0)

    # Test builtin cast
    integer_value = TestSSAValue(i32)
    cast_op = builtin.UnrealizedConversionCastOp.get(
        (integer_value,), result_type=(IndexType(),)
    )
    interpreter.register_implementations(BuiltinFunctions())

    results = interpreter.run_op(cast_op, (integer,))
    assert len(results) == 1
    index1 = results[0]
    assert isinstance(index1, Index)
