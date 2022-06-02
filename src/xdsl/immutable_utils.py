from xdsl.immutable_ir import *
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
from xdsl.immutable_ir import _unpack_operands


def new_cst(value: int, width: int = 32) -> List[IOp]:
    return new_op(
        arith.Constant,
        attributes={"value": IntegerAttr.from_int_and_width(value, width)},
        result_types=[IntegerType.from_width(width)])


def new_bin_op(op_type: Type[Operation],
               lhs: ISSAValue | IOp | List[IOp],
               rhs: ISSAValue | IOp | List[IOp],
               attributes: Optional[Dict[str, Attribute]] = None):
    """
    Generates a new op of `op_type`. Only to be used for ops which produce one 
    result and lhs.type == rhs.type == result.type
    """
    if attributes is None:
        attributes = {}
    lhs_unpacked, _ = _unpack_operands([lhs])
    return new_op(op_type,
                  operands=[lhs, rhs],
                  attributes=attributes,
                  result_types=[lhs_unpacked[-1].typ])


######################### more specialized #########################

# TODO: Do we want specialized stuff like this?


def new_cmpi(pred: str, lhs: ISSAValue | IOp | List[IOp],
             rhs: ISSAValue | IOp | List[IOp]):
    predicate: int = 0
    match pred:
        case "eq":
            predicate = 0
        case "ne":
            predicate = 1
        case "slt":
            predicate = 2
        case "sle":
            predicate = 3
        case "sgt":
            predicate = 4
        case "sge":
            predicate = 5
        case "ult":
            predicate = 6
        case "ule":
            predicate = 7
        case "ugt":
            predicate = 8
        case "uge":
            predicate = 9
        case _:
            raise Exception("Invalid predicate for op arith.cmpi")
    return new_op(arith.Cmpi,
                  operands=[lhs, rhs],
                  attributes={
                      "predicate":
                      IntegerAttr.from_int_and_width(predicate, 64)
                  },
                  result_types=[IntegerType.from_width(1)])
