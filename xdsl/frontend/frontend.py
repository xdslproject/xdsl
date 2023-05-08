import inspect
from typing import Dict, Any


# TODO: Clean this up after prototyping
class Frontend:
    map = {}

    def add_mapping(self, func, operation):
        # adds a mapping from a function to an operation
        self.map[func] = operation


def frontend_op(frontend: Frontend, operation):
    def decorator(func):
        frontend.add_mapping(func, operation)
        return func

    return decorator


def frontend_type(cls: Any):
    supported_functions = [
        # "__new__",
        # "__init__",
        # "__del__",
        # "__repr__",
        # "__str__",
        # "__format__",
        "__bytes__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__bool__",
        "__getattr__",
        # "__getattribute__",
        # "__setattr__",
        # "__delattr__",
        # "__dir__",
        "__call__",
        "__len__",
        "__length_hint__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__iter__",
        "__reversed__",
        "__contains__",
        "__add__",
        "__sub__",
        "__mul__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__divmod__",
        "__pow__",
        "__lshift__",
        "__rshift__",
        "__and__",
        "__xor__",
        "__or__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rmatmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        "__rdivmod__",
        "__rpow__",
        "__rlshift__",
        "__rrshift__",
        "__rand__",
        "__rxor__",
        "__ror__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__imatmul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__ilshift__",
        "__irshift__",
        "__iand__",
        "__ixor__",
        "__ior__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__complex__",
        "__int__",
        "__float__",
        "__index__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
        "__enter__",
        "__exit__",
        "__await__",
        "__aiter__",
        "__anext__",
        "__aenter__",
        "__aexit__",
        "__copy__",
        "__deepcopy__",
    ]

    magic_functions: Dict[str, str] = {}
    for name in dir(cls):
        attr = getattr(cls, name)
        if callable(attr) and name.startswith("__") and name.endswith("__"):
            if name in supported_functions:
                magic_functions[name] = inspect.getsource(attr)
    cls.magic_functions = magic_functions
    return cls
