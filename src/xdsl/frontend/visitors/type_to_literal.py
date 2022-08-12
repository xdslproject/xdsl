import builtins
import logging
from typing import *
from typing import _GenericAlias


class TypeToLiteralException(Exception):
    pass


_RetType = Union[bool, int, str, bytes, list, tuple]
_ArgType = Tuple[Type]


class TypeToLiteralVisitor:
    """
    Visitor class to convert type hints into raw literals.
    E.g., we convert
    Tuple[Literal[10], Literal[20]]
    to
    (10, 20)
    """

    def __init__(self, logger: Optional[logging.RootLogger] = None) -> None:
        if not logger:
            logger = logging.getLogger("type_to_literal_logger")
            logger.setLevel(logging.INFO)
        self.logger = logger

    def visit(self, typ: Type) -> _RetType:
        """Visit a type."""
        if isinstance(typ, _GenericAlias):
            method = "visit_" + typ.__origin__.__name__
            args = typ.__args__
        else:
            method = 'visit_' + typ.__class__.__name__
            args = None

        if not hasattr(self, method):
            raise TypeToLiteralException(
                f"No '{method}' function defined for {typ}")

        visitor = getattr(self, method)
        return visitor(args)

    def visit_list(self, args: _ArgType) -> _RetType:
        return [self.visit(arg) for arg in args]

    def visit_Literal(self, args: _ArgType) -> _RetType:
        if len(args) > 1:
            return self.visit_list(args)
        return args[0]

    def visit_Tuple(self, args: _ArgType) -> _RetType:
        return tuple(self.visit_list(args))

    def visit_Dict(self, args: _ArgType) -> _RetType:
        raise TypeToLiteralException(
            "Dictionary type cannot be converted to literal.")

    def visit_Union(self, args: _ArgType) -> _RetType:
        raise TypeToLiteralException(
            "Union type cannot be converted to literal.")

    def visit_Sequence(self, args: _ArgType) -> _RetType:
        raise TypeToLiteralException(
            "Sequence type cannot be converted to literal.")

    def visit_Callable(self, args: _ArgType) -> _RetType:
        raise TypeToLiteralException(
            "Callable type cannot be converted to literal.")
