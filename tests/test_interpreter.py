from dataclasses import dataclass

import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, register_impls


def test_import_functions():

    @dataclass
    class A(InterpreterFunctions):
        pass

    @register_impls
    @dataclass
    class B(InterpreterFunctions):
        pass

    i = Interpreter(ModuleOp.from_region_or_ops([]))

    i.register_implementations(B())

    with pytest.raises(ValueError) as e:
        i.register_implementations(A())

    assert e.value.args[0] == 'Use `@register_impls` on class A'
