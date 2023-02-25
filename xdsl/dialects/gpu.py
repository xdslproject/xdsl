from __future__ import annotations

from xdsl.ir import Operation, Region, Dialect
from xdsl.irdl import irdl_op_definition, SingleBlockRegion, OpAttr
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ModuleOp(Operation):
    name = "gpu.module"

    body: SingleBlockRegion
    sym_name: OpAttr[StringAttr]

    @staticmethod
    def get(name: SymbolRefAttr, body: Region) -> ModuleOp:
        op = ModuleOp.build(attributes={"sym_name": name}, regions=[body])
        return op

    def verify_(self):
        if (len(self.body.ops) > 0
                and not isinstance(self.body.ops[-1], ModuleEndOp)):
            raise VerifyException("gpu.module must end with gpu.module_end")


@irdl_op_definition
class ModuleEndOp(Operation):
    name = "gpu.module_end"

    @staticmethod
    def get() -> ModuleEndOp:
        return ModuleEndOp.build()


GPU = Dialect([ModuleOp, ModuleEndOp], [])
