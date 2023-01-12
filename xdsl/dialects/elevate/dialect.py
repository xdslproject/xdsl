from __future__ import annotations
from typing import Type
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *
import xdsl.elevate as elevate


@irdl_attr_definition
class OpHandle(ParametrizedAttribute):
    name = "op_handle"


@irdl_attr_definition
class StrategyHandle(ParametrizedAttribute):
    name = "strategy_handle"


@irdl_attr_definition
class Strategy(ParametrizedAttribute):
    name = "strategy"


@irdl_attr_definition
class RewriteResult(ParametrizedAttribute):
    name = "rewrite_result"


# Explicit application
class ElevateOperation(Operation, ABC):

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        pass


@irdl_op_definition
class StrategyStratOp(ElevateOperation):
    name: str = "elevate2.strategy"
    strategy: Annotated[Operand, StrategyHandle]
    output: Annotated[OpResult, StrategyHandle]
    body = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.id


@irdl_op_definition
class ApplyStratOp(ElevateOperation):
    name: str = "elevate2.apply"
    op: Annotated[Operand, OpHandle]
    strategy: Annotated[Operand, StrategyHandle]
    output: Annotated[OpResult, RewriteResult]

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.id


@irdl_op_definition
class IdStratOp(ElevateOperation):
    name: str = "elevate2.id"
    output: Annotated[OpResult, StrategyHandle]

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.id


@irdl_op_definition
class FailStratOp(ElevateOperation):
    name: str = "elevate2.fail"
    output: Annotated[OpResult, StrategyHandle]

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.fail


@irdl_op_definition
class SeqStratOp(ElevateOperation):
    name: str = "elevate2.seq"
    s0: Annotated[Operand, StrategyHandle]
    s1: Annotated[Operand, StrategyHandle]
    output: Annotated[OpResult, StrategyHandle]

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.seq


@irdl_op_definition
class TopToBottomStratOp(ElevateOperation):
    name: str = "elevate2.toptobottom"
    input: Annotated[Operand, StrategyHandle]
    output: Annotated[OpResult, StrategyHandle]

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.topToBottom


@irdl_op_definition
class TryStratOp(ElevateOperation):
    name: str = "elevate2.try"
    input: Annotated[Operand, StrategyHandle]
    output: Annotated[OpResult, StrategyHandle]

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.try_


@irdl_op_definition
class ReturnStratOp(ElevateOperation):
    name: str = "elevate2.return"
    input: Annotated[Operand, StrategyHandle]


# Implicit direct application
@irdl_op_definition
class StrategyOp(ElevateOperation):
    name: str = "elevate.strategy"
    output: Annotated[OpResult, Strategy]
    body = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.id


@irdl_op_definition
class ApplyOp(ElevateOperation):
    name: str = "elevate.apply"
    strategy: Annotated[Operand, Strategy]
    args = VarOperandDef(Attribute)
    body = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        raise Exception("unreachable")
        return elevate.id


@irdl_op_definition
class ComposeOp(ElevateOperation):
    name: str = "elevate.compose"
    strategy_name = AttributeDef(StringAttr)
    output: Annotated[OpResult, Strategy]
    body = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        raise Exception("unreachable")
        return elevate.id


@irdl_op_definition
class IdOp(ElevateOperation):
    name: str = "elevate.id"
    # input: Annotated[Operand, OpHandle)
    # output: Annotated[OpResult, OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.id


@irdl_op_definition
class FailOp(ElevateOperation):
    name: str = "elevate.fail"
    # input: Annotated[Operand, OpHandle)
    # output: Annotated[OpResult, OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.fail


@irdl_op_definition
class TopToBottomOp(ElevateOperation):
    name: str = "elevate.toptobottom"
    region = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.topToBottom


@irdl_op_definition
class BottomToTopOp(ElevateOperation):
    name: str = "elevate.bottomtotop"
    region = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.bottomToTop


@irdl_op_definition
class EverywhereOp(ElevateOperation):
    name: str = "elevate.everywhere"
    region = RegionDef()

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.everywhere


@irdl_op_definition
class TryOp(ElevateOperation):
    name: str = "elevate.try"
    region = RegionDef()
    # output: Annotated[OpResult, OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.try_


@irdl_op_definition
class RepeatNOp(ElevateOperation):
    name: str = "elevate.repeatN"
    region = RegionDef()
    # output: Annotated[OpResult, OpHandle)

    @classmethod
    def get_strategy(cls) -> Type[elevate.Strategy]:
        return elevate.repeatN


@irdl_op_definition
class ReturnOp(Operation):
    name: str = "elevate.return"
    # input: Annotated[Operand, OpHandle)


@irdl_op_definition
class NativeStrategyOp(Operation):
    name: str = "elevate.native"
    strategy_name = AttributeDef(StringAttr)
    strategy: Annotated[OpResult, Strategy]


Elevate = Dialect(
    [
        # Strategies which directly apply in the representation, i.e. produce an ophandle.
        StrategyOp,
        ComposeOp,
        ApplyOp,
        IdOp,
        FailOp,
        TopToBottomOp,
        BottomToTopOp,
        TryOp,
        RepeatNOp,
        ReturnOp,
        NativeStrategyOp,

        # Strategies with explicit application
        StrategyStratOp,
        ApplyStratOp,
        IdStratOp,
        FailStratOp,
        SeqStratOp,
        TopToBottomStratOp,
        EverywhereOp,
        TryStratOp,
        ReturnStratOp,
    ],
    [
        OpHandle,
        StrategyHandle,
        Strategy,
        RewriteResult,
    ])
