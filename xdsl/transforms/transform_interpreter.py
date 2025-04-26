"""Transform interpreter."""

from dataclasses import dataclass

from xdsl.dialects import builtin, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import TransformFunctions
from xdsl.passes import Context, ModulePass


@dataclass(frozen=True)
class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    # TODO: Add the rest of options to the transform interpreter
    # and add semantics for all of the other transform operations

    name = "transform-interpreter"

    entry_point: str = "__transform_main"

    @staticmethod
    def find_transform_entry_point(
        root: builtin.ModuleOp, entry_point: str
    ) -> transform.NamedSequenceOp:
        for op in root.walk():
            if (
                isinstance(op, transform.NamedSequenceOp)
                and op.sym_name.data == entry_point
            ):
                return op

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        schedule = TransformInterpreterPass.find_transform_entry_point(
            op, self.entry_point
        )
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctions())
        interpreter.call_op(schedule, (op,))
