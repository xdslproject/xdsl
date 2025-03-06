import dis
import inspect
import sys
from collections.abc import Callable
from types import CodeType
from typing import Any

from benchmarks.microbenchmarks import Extensibility


def trace_calls(func: Callable[..., Any], indent: int = 4):
    """Disassemble a function and all functions it calls."""
    # Keep track of functions we've already disassembled
    seen: set[CodeType] = set()

    def _trace_calls(
        f: Callable[
            [
                Any,
            ],
            Any,
        ],
        depth: int = 0,
    ):
        # Ignore already seen functions
        if f.__code__ in seen:
            return
        seen.add(f.__code__)

        print(f"\n{' ' * depth * indent}{'=' * 43}")
        print(f"{' ' * depth * indent}Disassembling `{f.__name__}`:")
        print(f"{' ' * depth * indent}{'=' * 43}")
        print(
            " " * depth * indent
            + ">>> "
            + f"\n{' ' * depth * indent}>>> ".join(inspect.getsource(f).split("\n"))
        )
        print(f"{' ' * depth * indent}{'=' * 43}")
        print(
            " " * depth * indent
            + "+"
            + f"\n{' ' * depth * indent}+".join(dis.Bytecode(f).dis().split("\n"))
        )

        # Find all global names used in this function
        for name in f.__code__.co_names:
            # Try to get the object from globals or builtins
            if name in f.__globals__:
                obj = f.__globals__[name]
                # If it's a function, disassemble it too
                if inspect.isfunction(obj):
                    _trace_calls(obj, depth=depth + 1)

        # Look for function references in default args and closure
        if f.__defaults__:
            for default in f.__defaults__:
                if inspect.isfunction(default):
                    _trace_calls(default, depth=depth + 1)

        if f.__closure__:
            for cell in f.__closure__:
                if inspect.isfunction(cell.cell_contents):
                    _trace_calls(cell.cell_contents, depth=depth + 1)

    _trace_calls(func, depth=0)


# def trace_execution_and_show_bytecode(func: Callable[..., Any], *args: Any, **kwargs: Any):
#     executed_functions = set()

#     def tracer(frame, event, arg):
#         if event == 'call':
#             func_obj = frame.f_globals.get(frame.f_code.co_name)
#             if func_obj and inspect.isfunction(func_obj):
#                 executed_functions.add(func_obj)
#                 print(func_obj.__name__)
#         return tracer

#     # Set the tracer
#     sys.settrace(tracer)

#     # Run the function
#     result = func()

#     # Turn off tracing
#     sys.settrace(None)

#     # Show bytecode for each executed function
#     print("\n=== Bytecode of Executed Functions ===")
#     indent = 4
#     depth = 0
#     for func_obj in executed_functions:
#         # print(f"\n{'='*40}")
#         # print(f"Disassembling {func_obj.__name__}:")
#         # print(f"{'='*40}")
#         # dis.dis(func_obj)
#         print(f"\n{' '*depth*indent}{'='*43}")
#         print(f"{' '*depth*indent}Disassembling `{func_obj.__name__}`:")
#         print(f"{' '*depth*indent}{'='*43}")
#         print(
#             ' '*depth*indent + ">>> "+
#             f"\n{' '*depth*indent}>>> ".join(inspect.getsource(func_obj).split("\n"))

#         )
#         print(f"{' '*depth*indent}{'='*43}")
#         print(
#             ' '*depth*indent + "+"+
#             f"\n{' '*depth*indent}+".join(dis.Bytecode(func_obj).dis().split("\n"))
#         )

#     return result


def trace_execution_and_show_bytecode(
    func: Callable[..., Any], *args: Any, **kwargs: Any
):
    executed_functions: set[Callable[..., Any]] = set()

    def tracer(frame, event, arg):
        if event == "call":
            func_obj = frame.f_globals.get(frame.f_code.co_name)
            if func_obj and inspect.isfunction(func_obj):
                executed_functions.add(func_obj)
        return tracer

    # Set the tracer
    sys.settrace(tracer)

    # Run the function
    result = func(*args, **kwargs)

    # Turn off tracing
    sys.settrace(None)

    # Show bytecode for each executed function
    print("\n// === Bytecode of Executed Functions ===")
    for func_obj in executed_functions:
        print(f"\n// {'=' * 40}")
        print(f"// Disassembling {func_obj.__name__}:")
        print(f"// {'=' * 40}")
        dis.dis(func_obj)

    return result


# Example usage
def helper_function(x: int) -> int:
    return x * 2


def main_function() -> int:
    result = helper_function(10)
    return result + 5


# trace_calls(main_function)
# trace_execution_and_show_bytecode(main_function)

EXTENSIBILITY = Extensibility()
# trace_calls(IR_TRAVERSAL.time_walk_block_ops)
trace_execution_and_show_bytecode(EXTENSIBILITY.time_trait_check)
