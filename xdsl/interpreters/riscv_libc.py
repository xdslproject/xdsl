from math import ceil
from typing import Any

from interpreters.memref import MemrefValue
from interpreters.riscv import Buffer

from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl_external,
    register_impls,
)
from xdsl.ir import Operation


@register_impls
class RiscvLibc(InterpreterFunctions):
    @impl_external("malloc")
    def malloc(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        assert len(args) == 1
        assert isinstance(args[0], int)
        size = args[0]
        if size % 4 != 0:
            # malloc a bit too much if not word-aligned
            size = ceil(size / 4) * 4

        # set values to 1 to signify uninitialized memory
        return (Buffer([MemrefValue.Uninitialized] * (size // 4)),)

    @impl_external("calloc")
    def calloc(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        assert len(args) == 2
        assert isinstance(args[0], int)
        assert isinstance(args[1], int)
        num = args[0]
        size = args[1]

        num_bytes = num * size

        if num_bytes % 4 != 0:
            # malloc a bit too much if not word-aligned
            num_bytes = ceil(num_bytes / 4) * 4

        return (Buffer([0] * (num_bytes // 4)),)

    @impl_external("free")
    def free(
        self, interpreter: Interpreter, op: Operation, args: PythonValues
    ) -> PythonValues:
        assert len(args) == 1
        assert isinstance(args[0], Buffer)

        buff: Buffer[Any] = args[0]
        for i in range(len(buff.data)):
            buff[i] = MemrefValue.Deallocated

        return tuple()

    @impl_external("putchar")
    def putchar(self, interp: Interpreter, op: Operation, args: PythonValues):
        assert len(args) == 1
        char = args[0]
        if isinstance(char, int):
            print(chr(char), end="")
        else:
            print(char, end="")
        return tuple()
