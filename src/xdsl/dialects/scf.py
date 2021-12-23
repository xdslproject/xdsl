from __future__ import annotations
from xdsl.dialects.builtin import *


@dataclass
class Scf:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(If)
        self.ctx.register_op(Yield)
        self.ctx.register_op(Condition)
        self.ctx.register_op(While)


@irdl_op_definition
class If(Operation):
    name: str = "scf.if"
    output = VarResultDef(AnyAttr())
    cond = OperandDef(IntegerType.from_width(1))

    true_region = RegionDef()
    # TODO this should be optional under certain conditions
    false_region = RegionDef()

    @staticmethod
    def get(cond: Union[Operation, SSAValue], return_types: List[Attribute],
            true_region: Union[Region, List[Block], List[Operation]],
            false_region: Union[Region, List[Block], List[Operation]]):
        return If.build(operands=[cond],
                        result_types=return_types,
                        regions=[true_region, false_region])


@irdl_op_definition
class Yield(Operation):
    name: str = "scf.yield"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*operands: Union[Operation, SSAValue]) -> Yield:
        return Yield.create(operands=[operand for operand in operands])


@irdl_op_definition
class Condition(Operation):
    name: str = "scf.condition"
    cond = OperandDef(IntegerType.from_width(1))
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(cond: Union[Operation, SSAValue],
            *output_ops: Union[Operation, SSAValue]) -> Condition:
        return Condition.build(
            operands=[cond, [output for output in output_ops]])


@irdl_op_definition
class While(Operation):
    name: str = "scf.while"
    arguments = VarOperandDef(AnyAttr())

    res = VarResultDef(AnyAttr())
    before_region = RegionDef()
    after_region = RegionDef()

    # TODO verify dependencies between scf.condition, scf.yield and the regions
    def verify_(self):
        for (idx, arg) in enumerate(self.arguments):
            if self.before_region.blocks[0].args[idx].typ != arg.typ:
                raise Exception(
                    f"Block arguments with wrong type, expected {arg.typ}, got {self.before_region.block[0].args[idx].typ}"
                )

        for (idx, res) in enumerate(self.res):
            if self.after_region.blocks[0].args[idx].typ != res.typ:
                raise Exception(
                    f"Block arguments with wrong type, expected {res.typ}, got {self.after_region.block[0].args[idx].typ}"
                )

    @staticmethod
    def get(operands: List[Union[Operation,
                                 SSAValue]], result_types: List[Attribute],
            before: Union[Region, List[Operation], List[Block]],
            after: Union[Region, List[Operation], List[Block]]) -> While:
        op = While.build(operands=operands,
                         result_types=result_types,
                         regions=[before, after])
        return op
