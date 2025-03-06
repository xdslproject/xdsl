import dis
import inspect
import sys
import textwrap
from collections.abc import Callable
from types import CodeType
from typing import Any

from benchmarks.microbenchmarks import Extensibility


def get_called_functions(
    func: Callable[[], Any], repeats: bool = True
) -> list[tuple[str, CodeType | None, int]]:
    """Get called functions and their stack depths."""
    names: list[CodeType] = []
    objs: list[CodeType | None] = []
    depths: list[int] = []
    stack: list[str] = []

    def trace_calls(frame, event, _arg):
        # TODO: event == opcode
        if event == "call":
            # print(names)
            # print(objs)
            # print(stack)
            # print()
            # Get the function name
            func_name = frame.f_code.co_name
            # print(func_name)

            # Get the function object (if it's a method, try to get it from locals)
            # func_obj = frame.f_globals.get(frame.f_code.co_name)
            # if func_obj is None and 'self' in frame.f_locals:
            #     instance = frame.f_locals['self']
            #     func_obj = getattr(instance.__class__, frame.f_code.co_name, None)
            # if not inspect.isfunction(func_obj):
            #     func_obj = None

            # print(type(frame))
            # print(frame)
            # print(dir(frame))
            # print(f"f_back: {frame.f_back} -> `{frame.f_back.f_code.co_name}`")
            # print(f"f_builtins: {frame.f_builtins} -> unhelpful")
            # print(f"f_code: {frame.f_code} -> {dir(frame.f_code)}")
            # print(f"f_globals: {frame.f_globals}")
            # print(f"f_lasti: {frame.f_lasti} -> index of last attempted instruction")
            # print(f"f_lineno: {frame.f_lineno}")
            # print(f"f_locals: {frame.f_locals}")
            # print(f"f_trace: {frame.f_trace}")
            # print(f"f_trace_lines: {frame.f_trace_lines}")
            # print(f"f_trace_opcodes: {frame.f_trace_opcodes}")
            # print("\n\n")

            func_obj = None

            # Check if it's a method of a class instance (instance method)
            if "self" in frame.f_locals:
                instance = frame.f_locals["self"]
                method = getattr(instance.__class__, func_name, None)
                if method and frame.f_code == getattr(method, "__code__", None):
                    func_obj = method

            # Check if it's a class method
            elif "cls" in frame.f_locals:
                cls = frame.f_locals["cls"]
                method = getattr(cls, func_name, None)
                # For class methods, we need to check the __func__ attribute
                if (
                    method
                    and hasattr(method, "__func__")
                    and frame.f_code == getattr(method.__func__, "__code__", None)
                ):
                    func_obj = method

            # Check for static methods
            if func_obj is None:
                caller_frame = frame.f_back
                if caller_frame:
                    # Check if the call was made through a class or instance
                    for value in caller_frame.f_locals.values():
                        if isinstance(value, type):  # It's a class
                            static_method = getattr(value, func_name, None)
                            # For static methods, the code object is directly attached
                            if (
                                static_method
                                and getattr(static_method, "__code__", None)
                                == frame.f_code
                            ):
                                func_obj = static_method
                                break
                        elif hasattr(value, "__class__") and not isinstance(
                            value, type
                        ):  # It's an instance
                            static_method = getattr(value.__class__, func_name, None)
                            if (
                                static_method
                                and getattr(static_method, "__code__", None)
                                == frame.f_code
                            ):
                                func_obj = static_method
                                break

                    # If not found yet, scan all classes in globals
                    if func_obj is None:
                        for value in frame.f_globals.values():
                            if isinstance(value, type):  # It's a class
                                static_method = getattr(value, func_name, None)
                                # Check if it's a static method by comparing code objects
                                if (
                                    static_method
                                    and getattr(static_method, "__code__", None)
                                    == frame.f_code
                                ):
                                    func_obj = static_method
                                    break

            # Check if it's a function in module globals
            if func_obj is None:
                for value in frame.f_globals.values():
                    if inspect.isfunction(value) and value.__code__ == frame.f_code:
                        func_obj = value
                        break

            # print(func_obj)

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


def print_bytecode(func: Callable[[], Any], indent: int = 4, repeats: bool = True):
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


# Example functions to trace
def a():
    b()
    b()
    (lambda: b())()


def b():
    x = Foo()
    Foo.c()
    Foo.d()
    x.e()


class Foo:
    @staticmethod
    def c():
        pass

    @classmethod
    def d(cls):
        pass

    def e(self):
        pass


EXTENSIBILITY = Extensibility()
print_bytecode(EXTENSIBILITY.time_trait_check)
# print_bytecode(a)
