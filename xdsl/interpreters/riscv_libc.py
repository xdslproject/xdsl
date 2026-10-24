from math import ceil

from xdsl.dialects.builtin import IntegerAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl_external,
    register_impls,
)
from xdsl.interpreters.utils import ptr
from xdsl.ir import Operation
from xdsl.utils.hints import isa


@register_impls
class RiscvLibcFunctions(InterpreterFunctions):
    @impl_external("malloc")
    def malloc(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        assert len(args) == 1
        assert isa(args[0], IntegerAttr)
        size = args[0].value.data
        if size % 4 != 0:
            # malloc a bit too much if not word-aligned
            size = ceil(size / 4) * 4

        # set values to 1 to signify uninitialized memory
        return (ptr.RawPtr.zeros(size),)

    @impl_external("calloc")
    def calloc(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        assert len(args) == 2
        assert isa(args[0], IntegerAttr)
        assert isa(args[1], IntegerAttr)
        num = args[0].value.data
        size = args[1].value.data

        num_bytes = num * size

        if num_bytes % 4 != 0:
            # malloc a bit too much if not word-aligned
            num_bytes = ceil(num_bytes / 4) * 4

        return (ptr.RawPtr.zeros(size),)

    @impl_external("free")
    def free(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        assert len(args) == 1
        assert isinstance(args[0], ptr.RawPtr)
        return ()

    @impl_external("putchar")
    def putchar(self, iterpreter: Interpreter, op: Operation, args: PythonValues):
        assert len(args) == 1
        char = args[0]
        if isa(char, IntegerAttr):
            iterpreter.print(chr(char.value.data), end="")
        else:
            iterpreter.print(char, end="")
        return (char,)
