from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO

from xdsl.dialects import arith, func, scf
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
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

        taken_names = set(self.variables.values())
        if val.name_hint is not None and val.name_hint not in taken_names:
            name = val.name_hint
        else:
            prefix = "v" if val.name_hint is None else val.name_hint

            name = f"{prefix}{self._counter}"
            self._counter += 1

            while name in taken_names:
                name = f"{prefix}{self._counter}"
                self._counter += 1

        self.variables[val] = name
        return name

    def mlir_type_to_csl_type(self, type_attr: Attribute) -> str:
        """
        Convert an MLR type to a csl type. CSL supports a very limited set of types:

        - integer types: i16, u16, i32, u32
        - float types: f16, f32
        - pointers: [*]f32
        - arrays: [64]f32

        This method does not yet support all the types and will be expanded as needed later.
        """
        match type_attr:
            case Float16Type():
                return "f16"
            case Float32Type():
                return "f32"
            case IntegerType(
                width=IntAttr(data=width),
                signedness=SignednessAttr(data=Signedness.UNSIGNED),
            ):
                return f"u{width}"
            case IntegerType(width=IntAttr(data=width)):
                return f"i{width}"
            case _:
                return f"<!unknown type {type_attr}>"

    def attribute_value_to_str(self, attr: Attribute) -> str:
        """
        Takes a value-carrying attribute (IntegerAttr, FloatAttr, etc.)
        and converts it to a csl expression representing that value literal (0, 3.14, ...)
        """
        match attr:
            case IntAttr(data=val):
                return str(val)
            case IntegerAttr(value=val):
                return str(val.data)
            case FloatAttr(value=val):
                return str(val.data)
            case _:
                return f"<!unknown value {attr}>"

    def attribute_type_to_str(self, attr: Attribute) -> str:
        """
        Takes a value-carrying attribute and (IntegerAttr, FloatAttr, etc.)
        and converts it to a csl expression representing the value's type (f32, u16, ...)
        """
        match attr:
            case IntAttr():
                return "<!indeterminate IntAttr type>"
            case IntegerAttr(type=(IntegerType() | IndexType()) as int_t):
                return self.mlir_type_to_csl_type(int_t)
            case FloatAttr(type=(Float16Type() | Float32Type()) as float_t):
                return self.mlir_type_to_csl_type(float_t)
            case _:
                return f"<!unknown type of {attr}>"

    def print_block(self, body: Block):
        """
        Walks over a block and prints every operation in the block.
        """
        for op in body.ops:
            match op:
                case arith.Constant(value=v, result=r):
                    # v is an attribute that "carries a value", e.g. an IntegerAttr or FloatAttr

                    # convert the attributes type to a csl type:
                    type_name = self.attribute_type_to_str(v)
                    # convert the carried value to a csl value
                    value_str = self.attribute_value_to_str(v)

                    # emit a constant instantiation:
                    self.print(
                        f"const {self._get_variable_name_for(r)} : {type_name} = {value_str};"
                    )
                case func.FuncOp(sym_name=name, body=bdy, function_type=ftyp) if len(
                    ftyp.inputs
                ) == 0 and len(ftyp.outputs) == 0:
                    # only functions without input / outputs supported for now.
                    self.print(f"fn {name.data}() {{")
                    self.descend().print_block(bdy.block)
                    self.print("}")
                case scf.For(lb=lower, ub=upper, step=stp, body=bdy):
                    idx, *_ = bdy.block.args
                    self.print(
                        f"for(@range({self.mlir_type_to_csl_type(idx.type)}, {self._get_variable_name_for(lower)}, {self._get_variable_name_for(upper)}, {self._get_variable_name_for(stp)})) |{self._get_variable_name_for(idx)}| {{"
                    )
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
    """
    Takes a module op and prints it to the given output stream.
    """
    ctx = CslPrintContext(output)
    ctx.print_block(prog.body.block)
