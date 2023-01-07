from __future__ import annotations
from typing import Annotated, Dict, List, Union
from dataclasses import dataclass
from xdsl.dialects.func import FuncOp

from xdsl.ir import Attribute, MLContext, OpResult, Region, SSAValue
from xdsl.irdl import Operand, irdl_op_definition, AttributeDef, AnyAttr, Operation
from xdsl.dialects.builtin import StringAttr, FlatSymbolRefAttr


@dataclass
class Symref:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Declare)
        self.ctx.register_op(Update)
        self.ctx.register_op(Fetch)


@irdl_op_definition
class Declare(Operation):
    name: str = "symref.declare"

    sym_name = AttributeDef(StringAttr)

    @staticmethod
    def get(sym_name: StringAttr) -> Declare:
        return Declare.build(attributes={"sym_name": sym_name})

    def _find_users(self, region: List[Region], uses: List[Union[Fetch, Update]]):
        symbol = self.attributes["sym_name"].data
        for block in region.blocks:
            for op in block.ops:
                if isinstance(op, Fetch) or isinstance(op, Update):
                    if symbol == op.attributes["symbol"].data.data:
                        uses.append(op)

                for region in op.regions:
                    self._find_users(region, uses)

    # TODO: we actually need to find users of all operations - this can be moved
    # to the parent class and `_find_users` adjusted.
    def users(self) -> List[Union[Fetch, Update]]:
        """Finds all users (updates and fetches) of this operation."""
        uses = []
        func: FuncOp = self.parent_op()
        for region in func.regions:
            self._find_users(region, uses)
        return uses


@irdl_op_definition
class Update(Operation):
    name: str = "symref.update"

    symbol = AttributeDef(FlatSymbolRefAttr)
    value: Annotated[Operand, AnyAttr()]

    @staticmethod
    def get(symbol: Union[str, FlatSymbolRefAttr], value: Operation | SSAValue) -> Update:
        return Update.build(operands=[value], attributes={"symbol": symbol})


@irdl_op_definition
class Fetch(Operation):
    name: str = "symref.fetch"

    symbol = AttributeDef(FlatSymbolRefAttr)
    value: Annotated[OpResult, AnyAttr()]

    @staticmethod
    def get(symbol: Union[str, FlatSymbolRefAttr], result_type: Attribute) -> Fetch:
        return Fetch.build(attributes={"symbol": symbol}, result_types=[result_type])
