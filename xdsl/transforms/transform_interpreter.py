"""Transform interpreter."""

from dataclasses import dataclass

from xdsl.dialects import builtin, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import TransformFunctions
from xdsl.passes import Context, ModulePass
from xdsl.transforms import get_all_passes
from xdsl.utils.exceptions import PassFailedException


@dataclass(frozen=True)
class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

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
        raise PassFailedException(
            f"{root} could not find a nested named sequence with name: {entry_point}"
        )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        schedule = TransformInterpreterPass.find_transform_entry_point(
            op, self.entry_point
        )
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctions(ctx, get_all_passes()))
        interpreter.call_op(schedule, (op,))
