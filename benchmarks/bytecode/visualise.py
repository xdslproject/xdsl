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
    return list(zip(names, objs, depths))


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
