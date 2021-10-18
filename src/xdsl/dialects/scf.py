from xdsl.dialects.builtin import *
from xdsl.util import OpOrBlockArg, get_ssa_value


@dataclass
class Scf:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(If)
        self.ctx.register_op(Yield)
        self.ctx.register_op(Condition)
        self.ctx.register_op(While)

    def if_(self, cond: OpOrBlockArg, true_region: Region,
            false_region: Region, return_types: List[Attribute]):
        op = Operation.with_result_types(If, [get_ssa_value(cond)],
                                         return_types, {})
        op.add_region(true_region)
        op.add_region(false_region)
        return op

    def yield_(self, *ops: OpOrBlockArg):
        return Operation.with_result_types(Yield,
                                           [get_ssa_value(op) for op in ops],
                                           [], {})

    def condition(self, cond: OpOrBlockArg, *output_ops: List[OpOrBlockArg]):
        return Operation.with_result_types(
            Condition,
            [get_ssa_value(cond)] + [get_ssa_value(op)
                                     for op in output_ops], [], {})

    def while_(self, before: Region, after: Region, ops: List[OpOrBlockArg],
               return_types: List[Attribute]):
        op = Operation.with_result_types(While,
                                         [get_ssa_value(op) for op in ops],
                                         return_types, {})
        op.add_region(before)
        op.add_region(after)
        return op


@irdl_op_definition
class If:
    name: str = "scf.if"
    output = VarResultDef(AnyAttr())
    cond = OperandDef(IntegerType.get(1))

    true_region = RegionDef()
    # TODO this should be optional under certain conditions
    false_region = RegionDef()


@irdl_op_definition
class Yield:
    name: str = "scf.yield"
    arguments = VarOperandDef(AnyAttr())


@irdl_op_definition
class Condition:
    name: str = "scf.condition"
    cond = OperandDef(IntegerType.get(1))
    arguments = VarOperandDef(AnyAttr())


@irdl_op_definition
class While:
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
