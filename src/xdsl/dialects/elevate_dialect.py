from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.elevate import *

@dataclass
class Elevate:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(OpHandle)
        self.ctx.register_attr(StrategyHandle)
        self.ctx.register_attr(RewriteResult)

        # Strategies which directly apply in the representation, i.e. produce an ophandle.
        self.ctx.register_op(StrategyOp)
        self.ctx.register_op(IdOp)
        self.ctx.register_op(FailOp)
        self.ctx.register_op(TopToBottomOp)
        self.ctx.register_op(TryOp)
        self.ctx.register_op(ReturnOp)


        # Strategies with explicit application
        self.ctx.register_op(StrategyStratOp)
        self.ctx.register_op(ApplyOp)
        self.ctx.register_op(IdStratOp)
        self.ctx.register_op(FailStratOp)
        self.ctx.register_op(SeqStratOp)
        self.ctx.register_op(TopToBottomStratOp)
        self.ctx.register_op(TryStratOp)
        self.ctx.register_op(ReturnStratOp)

@irdl_attr_definition
class OpHandle(ParametrizedAttribute):
    name = "op_handle"

@irdl_attr_definition
class StrategyHandle(ParametrizedAttribute):
    name = "strategy_handle"

@irdl_attr_definition
class RewriteResult(ParametrizedAttribute):
    name = "rewrite_result"

# Explicit application


class ElevateOperation(Operation, ABC):

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        pass

@irdl_op_definition
class StrategyStratOp(ElevateOperation):
    name: str = "elevate2.strategy"
    strategy = OperandDef(StrategyHandle)
    output = ResultDef(StrategyHandle)
    body = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return id

@irdl_op_definition
class ApplyOp(ElevateOperation):
    name: str = "elevate2.apply"
    op = OperandDef(OpHandle)
    strategy = OperandDef(StrategyHandle)
    output = ResultDef(RewriteResult)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return id


@irdl_op_definition
class IdStratOp(ElevateOperation):
    name: str = "elevate2.id"
    output = ResultDef(StrategyHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return id


@irdl_op_definition
class FailStratOp(ElevateOperation):
    name: str = "elevate2.fail"
    output = ResultDef(StrategyHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return fail


@irdl_op_definition
class SeqStratOp(ElevateOperation):
    name: str = "elevate2.seq"
    s0 = OperandDef(StrategyHandle)
    s1 = OperandDef(StrategyHandle)
    output = ResultDef(StrategyHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return seq


@irdl_op_definition
class TopToBottomStratOp(ElevateOperation):
    name: str = "elevate2.toptobottom"
    input = OperandDef(StrategyHandle)
    output = ResultDef(StrategyHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return topToBottom


@irdl_op_definition
class TryStratOp(ElevateOperation):
    name: str = "elevate2.try"
    input = OperandDef(StrategyHandle)
    output = ResultDef(StrategyHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return try_


@irdl_op_definition
class ReturnStratOp(ElevateOperation):
    name: str = "elevate2.return"
    input = OperandDef(StrategyHandle)


# Implicit direct application
@irdl_op_definition
class StrategyOp(ElevateOperation):
    name: str = "elevate.strategy"
    output = ResultDef(OpHandle)
    body = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return id


@irdl_op_definition
class IdOp(ElevateOperation):
    name: str = "elevate.id"
    input = OperandDef(OpHandle)
    output = ResultDef(OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return id




@irdl_op_definition
class FailOp(ElevateOperation):
    name: str = "elevate.fail"
    input = OperandDef(OpHandle)
    output = ResultDef(OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return fail


@irdl_op_definition
class TopToBottomOp(ElevateOperation):
    name: str = "elevate.toptobottom"
    region = RegionDef()
    output = ResultDef(OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return topToBottom


@irdl_op_definition
class TryOp(ElevateOperation):
    name: str = "elevate.try"
    region = RegionDef()
    output = ResultDef(OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[Strategy]:
        return try_


@irdl_op_definition
class ReturnOp(Operation):
    name: str = "elevate.return"
    input = OperandDef(OpHandle)