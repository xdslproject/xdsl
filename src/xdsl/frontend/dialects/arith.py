from typing import TypeVar

from xdsl.fronted.dialects.builtin import IntegerAttr, Float32Type, Float64Type

V = TypeVar("V", bound=int)


def Addi(input1: IntegerAttr[V], input2: IntegerAttr[V]) -> IntegerAttr[V]:
    """
    name: "arith.addi"

    Inputs:
        - input1: OperandDef(IntegerType)
        - input2: OperandDef(IntegerType)

    Returns:
        - output: ResultDef(IntegerType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from xdsl.dialects.arith import Addi
    return Addi  # type: ignore


U = TypeVar("U", bound=Float32Type | Float64Type)


def Addf(input1: U, input2: U) -> U:
    """
    TODO
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from xdsl.dialects.arith import Addf
    return Addf  # type: ignore


def Subf(input1: U, input2: U) -> U:
    """
    TODO
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from xdsl.dialects.arith import Subf
    return Subf  # type: ignore


def Mulf(input1: U, input2: U) -> U:
    """
    TODO
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from xdsl.dialects.arith import Mulf
    return Mulf  # type: ignore
