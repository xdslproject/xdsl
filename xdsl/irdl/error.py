from enum import Enum

from xdsl.ir import Operation
from xdsl.utils.diagnostic import Diagnostic


def error(op: Operation, msg: str, e: Exception):
    diag = Diagnostic()
    diag.add_message(op, msg)
    diag.raise_exception(msg, op, type(e), e)


class IRDLAnnotations(Enum):
    ParamDefAnnot = 1
    AttributeDefAnnot = 2
    OptAttributeDefAnnot = 3
    SingleBlockRegionAnnot = 4
    ConstraintVarAnnot = 5
