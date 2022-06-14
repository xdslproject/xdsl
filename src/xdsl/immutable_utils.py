from __future__ import annotations
from xdsl.immutable_ir import *
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
from xdsl.immutable_ir import _unpack_operands
from xdsl.elevate import *


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


@dataclass(frozen=True)
class GarbageCollect(Strategy):
    """
    Removes all unused operations from the regions nested under the op this is applied to. 
    Currently does not take ops with side effects into account
    """

    def impl(self, op: IOp) -> RewriteResult:
        # TODO: None here is very inelegant
        return regionsTopToBottom(blocksTopToBottom(self.GarbageCollectOps(None))).apply(op)

    @dataclass(frozen=True)
    class GarbageCollectOps(OpsTraversal):
        start_index: int = 0

        # This is exactly AllOpsTopToBottom, but the strategy requires the `new_block` as argument.
        # This could be implemented way more efficiently, but this is much more concise
        def impl(self, block: IBlock) -> Optional[IBlock]:
            if self.start_index == len(block.ops):
                # We visited all ops successfully previously
                return block
            new_block = opN(self.GarbageCollectOp(block), self.start_index).apply(block)
            if new_block is None:
                return None
            # To not miss any ops when the block is modified, we advance by ops_added +1.
            # i.e. we skip newly created ops and avoid skipping an op if the matched op was deleted
            ops_added = len(new_block.ops) - len(block.ops)
        
            # This should be GarbageCollectOps(... but for some reason this does not work for me
            return self.__class__(self.GarbageCollectOp(new_block), start_index=self.start_index+1+ops_added).apply(new_block)

        @dataclass(frozen=True)
        class GarbageCollectOp(Strategy):
            parent_block: IBlock

            def impl(self, op: IOp) -> RewriteResult:
                matched_op_used = False
                # If op does not have any results, there can't be uses.
                # Continue GC inside the regions of op
                if len(op.results) == 0:
                    if len(op.regions) > 0:
                        return try_(GarbageCollect()).apply(op)
                    return success(op)


                def uses_matched_op(sib_op: IOp):
                                    for result in op.results:
                                        nonlocal matched_op_used
                                        if result in sib_op.operands:
                                            # nonlocal matched_op_used
                                            matched_op_used = True
                for sibling_op in self.parent_block.ops:
                    sibling_op.walk(uses_matched_op)

                if not matched_op_used:
                    # empty replacement for the unused op
                    return success([])
                # Op has uses, continue GC in the regions of the op
                if len(op.regions) > 0:
                        return try_(GarbageCollect()).apply(op)
                return success(op)


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
