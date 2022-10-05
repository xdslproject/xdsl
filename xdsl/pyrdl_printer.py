from dataclasses import dataclass
from io import IOBase
from typing import Any
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.irdl import DialectOp

INDENTATION_SIZE: int = 4
"""The number of spaces per identation level"""


@dataclass
class PyRDLPrinter:

    stream: IOBase
    """The stream where we output the dialect definitions"""

    def _print(self, *args: Any, end: str = '\n') -> None:
        print(*args, file=self.stream, end=end)

    @staticmethod
    def snake_case_to_pascal_case(name: str) -> str:
        """Convert a snake_case name to PascalCase."""
        return ''.join([part.capitalize() for part in name.split('_')])

    def print_module(self, module: ModuleOp) -> None:
        """Print all dialect definitions in a module to pyrdl."""
        for op in module.ops:
            if isinstance(op, DialectOp):
                self.print_dialect(op)

    def print_dialect(self, dialect: DialectOp) -> None:
        """Convert the dialect definition to pyrdl."""
        dialect_py_name = self.snake_case_to_pascal_case(
            dialect.dialect_name.data)
        self._print('@dataclass')
        self._print(f'class {dialect_py_name}:')

        op_py_names = [
            self.snake_case_to_pascal_case(op.op_name.data)
            for op in dialect.get_op_defs()
        ]

        self._print(' ' * INDENTATION_SIZE, 'ctx: MLContext')
        self._print('\n')
        self._print(' ' * INDENTATION_SIZE, 'def __post_init__(self):')
        for op_py_name in op_py_names:
            self._print(' ' * INDENTATION_SIZE * 2,
                        f'self.ctx.register_op({op_py_name})')