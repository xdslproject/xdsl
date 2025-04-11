'''A tool to recursively visualise the bytecode for a function trace.

Example use case:
```text
$ python3 benchmarks/microbenchmarks.py IRTraversal.iterate_block_ops dis

// ===========================================
// Disassembling `time_iterate_block_ops`:
// ===========================================
// >>> def time_iterate_block_ops(self) -> None:
// >>>     """Time directly iterating over the linked list of a block's operations.
// >>>
// >>>     Comparison with `for (Operation &op : *block) {` at 2.15ns/op.
// >>>     """
// >>>     for op in IRTraversal.EXAMPLE_BLOCK.ops:
// >>>         assert op
// >>>
// ===========================================
 54           0 RESUME                   0

 59           2 LOAD_GLOBAL              0 (IRTraversal)
             12 LOAD_ATTR                2 (EXAMPLE_BLOCK)
             32 LOAD_ATTR                4 (ops)
             52 GET_ITER
        >>   54 FOR_ITER                 6 (to 70)
             58 STORE_FAST               1 (op)

 60          60 LOAD_FAST                1 (op)
             62 POP_JUMP_IF_FALSE        1 (to 66)
             64 JUMP_BACKWARD            6 (to 54)
        >>   66 LOAD_ASSERTION_ERROR
             68 RAISE_VARARGS            1

 59     >>   70 END_FOR
             72 RETURN_CONST             1 (None)


    // ===========================================
    // Disassembling `ops`:
    // ===========================================
    >>> # Unable to retrieve function body!

        // ===========================================
        // Disassembling `__init__`:
        // ===========================================
        >>> # Unable to retrieve function body!

    // ===========================================
    // Disassembling `__iter__`:
    // ===========================================
    // >>> def __iter__(self):
    // >>>     return _BlockOpsIterator(self.first)
    // >>>
    // ===========================================
1364           0 RESUME                   0

1365           2 LOAD_GLOBAL              1 (NULL + _BlockOpsIterator)
              12 LOAD_FAST                0 (self)
              14 LOAD_ATTR                2 (first)
              34 CALL                     1
              42 RETURN_VALUE


        // ===========================================
        // Disassembling `first`:
        // ===========================================
        >>> # Unable to retrieve function body!

            // ===========================================
            // Disassembling `first_op`:
            // ===========================================
            >>> # Unable to retrieve function body!

    // ===========================================
    // Disassembling `__next__`:
    // ===========================================
    // >>> def __next__(self):
    // >>>     next_op = self.next_op
    // >>>     if next_op is None:
    // >>>         raise StopIteration
    // >>>     self.next_op = next_op.next_op
    // >>>     return next_op
    // >>>
    // ===========================================
1327           0 RESUME                   0

1328           2 LOAD_FAST                0 (self)
               4 LOAD_ATTR                0 (next_op)
              24 STORE_FAST               1 (next_op)

1329          26 LOAD_FAST                1 (next_op)
              28 POP_JUMP_IF_NOT_NONE     6 (to 42)

1330          30 LOAD_GLOBAL              2 (StopIteration)
              40 RAISE_VARARGS            1

1331     >>   42 LOAD_FAST                1 (next_op)
              44 LOAD_ATTR                0 (next_op)
              64 LOAD_FAST                0 (self)
              66 STORE_ATTR               0 (next_op)

1332          76 LOAD_FAST                1 (next_op)
              78 RETURN_VALUE


        // ===========================================
        // Disassembling `next_op`:
        // ===========================================
        >>> # Unable to retrieve function body!
```
'''

import dis
import inspect
import sys
import textwrap
from collections.abc import Callable
from types import CodeType
from typing import Any


def get_called_functions(
    func: Callable[[], Any], repeats: bool = True
) -> list[tuple[str, CodeType | None, int]]:
    """Get called functions and their stack depths."""
    names: list[CodeType] = []
    objs: list[CodeType | None] = []
    depths: list[int] = []
    stack: list[str] = []

    def trace_calls(frame, event, _arg):
        if event == "call":
            # Get the function name
            func_name = frame.f_code.co_name

            # Get the function object (if it's a method, try to get it from locals)
            func_obj = frame.f_globals.get(frame.f_code.co_name)
            if func_obj is None and "self" in frame.f_locals:
                instance = frame.f_locals["self"]
                func_obj = getattr(instance.__class__, frame.f_code.co_name, None)
            if not inspect.isfunction(func_obj):
                func_obj = None

            if repeats or func_name not in names:
                names.append(func_name)
                objs.append(func_obj)
                depths.append(len(stack))
            stack.append(func_name)

        elif event == "return":
            stack.pop()
        return trace_calls

    sys.settrace(trace_calls)
    func()
    sys.settrace(None)
    return list(zip(names, objs, depths, strict=True))


def print_bytecode(func: Callable[[], Any], indent: int = 4, repeats: bool = False):
    """Disassemble a function and all functions it calls."""
    for name, obj, depth in get_called_functions(func, repeats=repeats):
        print(f"{' ' * depth * indent}// {'=' * 43}")
        print(f"{' ' * depth * indent}// Disassembling `{name}`:")
        print(f"{' ' * depth * indent}// {'=' * 43}")
        try:
            if not obj:
                raise OSError("could not get source code")
            source = textwrap.dedent(inspect.getsource(obj))
            print(
                " " * depth * indent
                + "// >>> "
                + f"\n{' ' * depth * indent}// >>> ".join(source.split("\n"))
            )
            print(f"{' ' * depth * indent}// {'=' * 43}")
            bytecode = dis.Bytecode(obj).dis()
            print(bytecode)
        except OSError:
            print(" " * depth * indent + ">>> # Unable to retrieve function body!")
        print()
