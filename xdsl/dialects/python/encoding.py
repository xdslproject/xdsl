"""
Helpers for encoding modules in the `python` dialect to Python source code.
"""

import abc
import dataclasses


@dataclasses.dataclass
class PythonSourceEncodingContext:
    name: str
    curr_id: int = 0

    def get_id(self):
        res = self.curr_id
        self.curr_id += 1
        return f"_{res}"

    def peek_id(self, offset: int):
        return f"_{self.curr_id - offset}"


class EncodingException(Exception): ...


class PythonSourceEncodable(abc.ABC):
    @abc.abstractmethod
    def encode(self, ctx: PythonSourceEncodingContext) -> list[str]:
        raise NotImplementedError()
