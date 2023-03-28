from abc import abstractmethod
from xdsl.ir import MLContext, Operation


class OperationPass:

    name: str

    @abstractmethod
    def apply(self, ctx: MLContext, op: Operation) -> None:
        pass
