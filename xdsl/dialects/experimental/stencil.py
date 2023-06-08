from __future__ import annotations

from typing import Sequence, TypeVar, cast, Iterable, Iterator

from operator import add, lt, neg

from xdsl.dialects import builtin
from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntAttr,
    ParametrizedAttribute,
    ArrayAttr,
    AnyFloat,
)
from xdsl.ir import Attribute, Operation, Dialect, TypeAttribute
from xdsl.ir import SSAValue

from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    Attribute,
    Region,
    VerifyException,
    Generic,
    Annotated,
    Operand,
    OpAttr,
    OpResult,
    VarOperand,
    VarOpResult,
    Block,
    IRDLOperation,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.hints import isa
