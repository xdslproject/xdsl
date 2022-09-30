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


def new_cst_with_type(value: int, type: Attribute) -> List[IOp]:
    if isinstance(type, IndexType):
        new_value = IntegerAttr.from_index_int_value(value)
    elif isinstance(type, IntegerType):
        new_value = IntegerAttr.from_int_and_width(value, type.width.data)
    else:
        raise Exception("Invalid type for constant")

    return new_op(
        arith.Constant,
        attributes={"value": new_value},
        result_types=[type])


def new_bin_op(op_type: Type[Operation],
               lhs: ISSAValue | IOp | List[IOp],
               rhs: ISSAValue | IOp | List[IOp],
               attributes: Optional[dict[str, Attribute]] = None):
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

def from_block(old_block: IBlock,
            args: Optional[Sequence[IBlockArg]] = None,
            ops: Optional[Sequence[IOp]] = None,
            env: Optional[dict[ISSAValue, ISSAValue]] = None,
            modify_op: Optional[Callable[[IOp, Optional[dict[ISSAValue, ISSAValue]]], List[IOp]]] = None) -> IBlock:
    """Creates a new iblock by assuming all fields of `old_block`, apart from 
    those specified via the arguments. 
    If new args are specified, they match the old args in length and no env is provided we add 
    an automatic mapping from the old_args to the new args.
    If `env` is specified all block_args and operands will be updated if they are included in
    the mapping.
    """

    if args is None:
        args = old_block.args

    if ops is None:
        ops = old_block.ops

    match (env, modify_op):
        case (None, None):
            new_ops: Sequence[IOp] = ops
        case (env, None):
            new_ops: Sequence[IOp] = []
            for op in ops:
                new_ops.extend(from_op(op, env=env))
        case (None, modify_op):
            new_ops: Sequence[IOp] = []
            for op in ops:
                new_ops.extend(modify_op(op, None))
        case (env, modify_op):
            new_ops: Sequence[IOp] = []
            for op in ops:
                new_ops.extend(modify_op(op, env))

    return IBlock(args=args, ops=new_ops)


def new_block(args: Optional[Sequence[IBlockArg] | Sequence[Attribute]] = None, 
            ops: Optional[List[IOp]] = None, 
            env: Optional[dict[ISSAValue, ISSAValue]] = None, 
            modify_op: Optional[Callable[[IOp, Optional[dict[ISSAValue, ISSAValue]]], List[IOp]]] = None) -> IBlock:
    """
    """
    if args is None:
        args = []
    if ops is None:
        ops = []

    match (env, modify_op):
        case (None, None):
            new_ops: Sequence[IOp] = ops
        case (env, None):
            new_ops: Sequence[IOp] = []
            for op in ops:
                new_ops.extend(from_op(op, env=env))
        case (None, modify_op):
            new_ops: Sequence[IOp] = []
            for op in ops:
                new_ops.extend(modify_op(op, None))
        case (env, modify_op):
            new_ops: Sequence[IOp] = []
            for op in ops:
                new_ops.extend(modify_op(op, env))

    return IBlock(args=args, ops=new_ops)


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


def new_cmpi(mnemonic: str, lhs: ISSAValue | IOp | List[IOp],
             rhs: ISSAValue | IOp | List[IOp]):
    predicate: int = 0
    match mnemonic:
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

def new_cmpf(mnemonic: str, lhs: ISSAValue | IOp | List[IOp],
             rhs: ISSAValue | IOp | List[IOp]):
    predicate: int = 0
    match mnemonic:
        case "false":
            predicate: int = 0
        case "oeq":
            predicate: int = 1
        case "ogt":
            predicate: int = 2
        case "oge":
            predicate: int = 3
        case "olt":
            predicate: int = 4
        case "ole":
            predicate: int = 5
        case "one":
            predicate: int = 6
        case "ord":
            predicate: int = 7
        case "ueq":
            predicate: int = 8
        case "ugt":
            predicate: int = 9
        case "uge":
            predicate: int = 10
        case "ult":
            predicate: int = 11
        case "ule":
            predicate: int = 12
        case "une":
            predicate: int = 13
        case "uno":
            predicate: int = 14
        case "true":
            predicate: int = 15
        case _:
            raise VerifyException(f"unknown cmpf mnemonic: {mnemonic}")
    if isinstance(lhs, list):
        typ = lhs[0].result.typ
    elif isinstance(lhs, IOp):
        typ = lhs.result.typ
    else:
        typ = lhs.typ

    if isinstance(typ, VectorType):
        result_type = VectorType.from_type_and_list(IntegerType.from_width(1), lhs.typ.get_shape())
    else:
        result_type = IntegerType.from_width(1)
    return new_op(arith.Cmpf,
                  operands=[lhs, rhs],
                  attributes={
                      "predicate":
                      IntegerAttr.from_int_and_width(predicate, 64)
                  },
                  result_types=[result_type])