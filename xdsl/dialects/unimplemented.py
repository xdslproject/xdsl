from dataclasses import dataclass
from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.irdl import AnyAttr, OperandDef, ResultDef, irdl_op_definition


@dataclass
class Arith:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Cast)
        self.ctx.register_op(TensorGenerate)


@irdl_op_definition
class TensorGenerate(Operation):
    name: str = "unimplemented.tensor.generate"
    result = ResultDef(AnyAttr())

    @staticmethod
    def get(result_type: Attribute) -> 'TensorGenerate':
        return TensorGenerate.create(result_types=[result_type])


@irdl_op_definition
class Cast(Operation):
    name: str = "unimplemented.cast"
    operand = OperandDef(AnyAttr())
    result = ResultDef(AnyAttr())

    @staticmethod
    def get(operand: Operation | SSAValue, result_type: Attribute) -> 'Cast':
        return Cast.build(operands=[operand], result_types=[result_type])
