import dataclasses

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.python import PythonFunctionOp
from xdsl.passes import ModulePass


@dataclasses.dataclass(frozen=True)
class PythonSerializeBytecodePass(ModulePass):
    name = "python-serialize-bytecode"

    output_path: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        with open(self.output_path, "wb") as fp:
            for inner in op.body.ops:
                if isinstance(inner, PythonFunctionOp):
                    inner.dump_to_file_as_single_module(fp)
