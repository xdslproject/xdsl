from __future__ import annotations
from typing import Generic, Type, TypeVar

from xdsl.ir import Attribute, Operation, Dialect, ParametrizedAttribute
from xdsl.irdl import ParameterDef, irdl_op_definition, irdl_attr_definition, SingleBlockRegion, OpAttr
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class _AllReduceOperationAttr(ParametrizedAttribute):
    name = "all_reduce_op"

    param: ParameterDef[StringAttr]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f"all_reduce_op {self.param.data}")


@irdl_attr_definition
class _DimensionAttr(ParametrizedAttribute):
    name = "dim"

    param: ParameterDef[StringAttr]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f"dim {self.param.data}")


T = TypeVar('T',
            bound=_AllReduceOperationAttr | _DimensionAttr,
            covariant=True)


@irdl_attr_definition
class _GPUAttr(ParametrizedAttribute, Generic[T]):
    name = "gpu"

    value: ParameterDef[T]

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_characters("<", f"Expected <")
        ntok = parser.tokenizer.next_token()

        if ntok.text == "dim":
            attrtype = _DimensionAttr
            vtok = parser.tokenizer.next_token()
            if vtok.text not in ["x", "y", "z"]:
                parser.raise_error(
                    f"Unexpected dim {vtok.text}. A gpu dim can only be x, y, or z",
                    vtok)

        elif ntok.text == "all_reduce_op":
            attrtype = _AllReduceOperationAttr
            vtok = parser.tokenizer.next_token()
            if vtok.text not in [
                    "add", "and", "max", "min", "mul", "or", "xor"
            ]:
                parser.raise_error(
                    f"Unexpected op {vtok.text}. A gpu all_reduce_op can only be add, "
                    "and, max, min, mul, or, or xor ", vtok)
        else:
            parser.raise_error(
                f"Unexpected token {ntok.text}. Expected dim or all_reduce_op",
                ntok)
        parser.parse_characters(">", f"Expected >")
        return [attrtype([StringAttr.from_str(vtok.text)])]

    @classmethod
    def from_op(cls: Type[_GPUAttr[_AllReduceOperationAttr]],
                value: str) -> AllReduceOperationAttr:
        return AllReduceOperationAttr(
            [_AllReduceOperationAttr([StringAttr(value)])])

    @classmethod
    def from_dimension(cls: Type[_GPUAttr[_DimensionAttr]],
                       value: str) -> DimensionAttr:
        return DimensionAttr([_DimensionAttr([StringAttr(value)])])

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        self.value.print_parameters(printer)
        printer.print_string(">")


DimensionAttr = _GPUAttr[_DimensionAttr]
AllReduceOperationAttr = _GPUAttr[_AllReduceOperationAttr]


@irdl_op_definition
class ModuleOp(Operation):
    name = "gpu.module"

    body: SingleBlockRegion
    sym_name: OpAttr[StringAttr]

    @staticmethod
    def get(name: SymbolRefAttr, ops: list[Operation]) -> ModuleOp:
        op = ModuleOp.build(attributes={"sym_name": name}, regions=[ops])
        return op

    def verify_(self):
        if (len(self.body.ops) == 0
                or not isinstance(self.body.ops[-1], ModuleEndOp)):
            raise VerifyException("gpu.module must end with gpu.module_end")


@irdl_op_definition
class ModuleEndOp(Operation):
    name = "gpu.module_end"

    @staticmethod
    def get() -> ModuleEndOp:
        return ModuleEndOp.build()


GPU = Dialect([ModuleOp, ModuleEndOp], [_GPUAttr])
