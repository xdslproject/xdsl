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
    # magic_functions: Dict[str, str] = {}
    # for name in dir(cls):
    #     attr = getattr(cls, name)
    #     if callable(attr) and name.startswith("__") and name.endswith("__"):
    #         magic_functions[name] = inspect.getsource(attr)
    # cls.overloads = magic_functions
    return cls
