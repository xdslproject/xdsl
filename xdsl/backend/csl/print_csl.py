from dataclasses import dataclass, field
from typing import IO

from xdsl.dialects import arith
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, SSAValue


@dataclass
class CslPrintContext:
    output: IO[str]

    variables: dict[SSAValue, str] = field(default_factory=dict)

    _counter: int = field(default=0)

    def print(self, text: str):
        print(text, file=self.output)

    def _get_variable_name_for(self, val: SSAValue) -> str:
        if val in self.variables:
            return self.variables[val]
        # generate a new variable name:
        name = f"v{self._counter}"
        self._counter += 1
        self.variables[val] = name
        return name

    def print_block(self, body: Block):
        for op in body.ops:
            match op:
                case arith.Constant(value=v, result=r):
                    self.print(
                        f"const {self._get_variable_name_for(r)} : {v.type} = {v.value.data};"
                    )
                case anyop:
                    self.print(f"// unknown op {anyop}")


def print_to_csl(prog: ModuleOp, output: IO[str]):
    ctx = CslPrintContext(output)
    ctx.print_block(prog.body.block)
