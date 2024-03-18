from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO

from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    IndexType,
    IntAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    SignednessAttr,
)
from xdsl.ir import Attribute, Block, SSAValue


@dataclass
class CslPrintContext:
    output: IO[str]

    variables: dict[SSAValue, str] = field(default_factory=dict)

    _counter: int = field(default=0)

    _prefix: str = field(default="")

    def print(self, text: str, prefix: str = ""):
        """
        Print `text` line by line, prefixed by self._prefix and prefix.
        """
        for l in text.split("\n"):
            print(self._prefix + prefix + l, file=self.output)

    def _get_variable_name_for(self, val: SSAValue) -> str:
        """
        Get an assigned variable name for a given SSA Value
        """
        if val in self.variables:
            return self.variables[val]

        prefix = "v"
        if val.name_hint is not None:
            prefix = val.name_hint

        taken_names = set(self.variables.values())
        if prefix not in taken_names:
            name = prefix
        else:
            name = f"{prefix}{self._counter}"
            self._counter += 1

            while name in taken_names:
                name = f"{prefix}{self._counter}"
                self._counter += 1

        self.variables[val] = name
        return name

    def mlir_type_to_csl_type(self, t: Attribute) -> str:
        match t:
            case Float16Type():
                return "f16"
            case Float32Type():
                return "f32"
            case IndexType():
                return "i64"
            case IntegerType(
                width=IntAttr(data=width),
                signedness=SignednessAttr(data=Signedness.UNSIGNED),
            ):
                return f"u{width}"
            case IntegerType(width=IntAttr(data=width)):
                return f"i{width}"
            case unkn:
                return f"!unsupported type: {unkn}"

    def print_block(self, body: Block):
        for op in body.ops:
            match op:
                case arith.Constant(value=v, result=r):
                    type_name = self.mlir_type_to_csl_type(
                        v.type  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                    )
                    value_str = f"{v.value.data}"  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

                    self.print(
                        f"const {self._get_variable_name_for(r)} : {type_name} = {value_str};"
                    )
                case func.FuncOp(sym_name=name, body=bdy) as funcop:
                    if len(funcop.function_type.inputs) > 0:
                        print("// can't print function with arguments")
                        continue
                    if len(funcop.function_type.outputs) > 0:
                        print("// can't print function with results")
                        continue

                    self.print(f"fn {name.data}() {{")
                    self.descend().print_block(bdy.block)
                    self.print("}")

                case anyop:
                    self.print(f"unknown op {anyop}", prefix="//")

    def descend(self) -> CslPrintContext:
        """
        Get a sub-context for descending into nested structures.

        Variables defined outside are valid inside, but inside varaibles will be
        available outside.
        """
        return CslPrintContext(
            output=self.output,
            variables=self.variables.copy(),
            _counter=self._counter,
            _prefix=self._prefix + "  ",
        )


def print_to_csl(prog: ModuleOp, output: IO[str]):
    ctx = CslPrintContext(output)
    ctx.print_block(prog.body.block)
