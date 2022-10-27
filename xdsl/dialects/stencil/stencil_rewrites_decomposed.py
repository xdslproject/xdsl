from __future__ import annotations
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil.stencil as stencil
import xdsl.dialects.builtin as builtin
import xdsl.dialects.func as func
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *

from io import StringIO
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil.stencil as stencil
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
from xdsl.dialects.stencil.stencil_inlining import InlineProducer, RerouteOutputDependency, RerouteInputDependency
import difflib

@dataclass(frozen=True)
class RemoveUnusedApplyOperands(Strategy):
    """
    Matches a stencil.apply operation where one of the block args is not used inside the region and 
    cleans up the operands and block_args
    """
    def impl(self, op: IOp) -> RewriteResult:
        match apply_op := op:
            # match for an apply where some of the block_args are not in the operands of any of the operations inside its region 
            case IOp(op_type=stencil.Apply,
                    regions=[IRegion(blocks=[block]) as region]) if any((args_unused := [not region.value_used_inside(arg) for arg in block.args])):
                new_operands = [operand for idx, operand in enumerate(apply_op.operands) if not args_unused[idx]]
                new_block_args = [arg for idx, arg in enumerate(block.args) if not args_unused[idx]]

                result = from_op(apply_op, operands=new_operands, regions=[IRegion(blocks=[from_block(block, args=new_block_args)])])
                return success(result)
            case _:
                return failure(self)


@dataclass(frozen=True)
class RemoveDuplicateApplyOperands(Strategy):
    """
    Matches a stencil.apply operation where one of the block args is not used inside the region and 
    cleans up the operands and block_args
    """
    def impl(self, op: IOp) -> RewriteResult:
        match apply_op := op:
            # match for an apply where some of the block_args are not in the operands of any of the operations inside its region 
            case IOp(op_type=stencil.Apply, operands=operands,
                    regions=[IRegion(blocks=[block])]) if any(((fst_occurence := operand1) == (snd_occurence := operand2) and (fst_idx := idx1) != (snd_idx := idx2)) for idx1, operand1 in enumerate(operands) for idx2, operand2 in enumerate(operands)):
                print("found an apply with duplicate operands")

                new_block_args: list[IBlockArg] = [arg for idx, arg in enumerate(block.args) if not idx == snd_idx]
                new_operands: list[ISSAValue] = apply_op.operands.copy()
                new_operands.remove(snd_occurence)

                result = from_op(apply_op, operands=new_operands, regions=[IRegion(blocks=[from_block(block, args=new_block_args, env={block.args[snd_idx]:block.args[fst_idx]})])])
                return success(result)
            case _:
                return failure(self)

def match_inlinable(consumer_apply: IOp) -> Optional[tuple[IOp, IOp, IResult]]:
    # Workaround until the registration and passing of values in native matchers is fixed
    if isinstance(consumer_apply, IResult):
        consumer_apply = consumer_apply.op
    # Explaining the matching:
    # - We match an apply op with operands and a region
    # - We check that one of the operands is another applyOp and remember it by `producer_apply`
    # - We check that inside the region is an AccessOp, that has the corresponding blockArg as operand and we remember it by `access_op_to_inline_at`
    # - we check that no empty stores are in the consumer apply + that no dynAccesses are in the consumer apply accessing the producer
    if consumer_apply.region is None or consumer_apply.region.block is None:
        return None

    for idx, operand in enumerate(consumer_apply.operands):
        # check whether operand is an Apply
        if isinstance((operand_to_inline := operand), IResult) and operand_to_inline.op.op_type == stencil.Apply:
            producer_apply = operand_to_inline.op
            # check whether the blockArg associated with the producer is used inside the region of the consumer
            access_op_to_inline_at: Optional[IOp] = None
            for op in consumer_apply.region.ops:
                if op.op_type == stencil.Access and consumer_apply.region.block.args[idx] in op.operands:
                    access_op_to_inline_at = op
                    break
            if not access_op_to_inline_at:
                continue

            # check whether no empty stores are in the consumer apply
            if any(op.op_type == stencil.Store and len(op.operands) == 0 for op in consumer_apply.region.ops):
                continue

            # check whether no dynAccesses are in the consumer apply accessing the producer
            if any(op.op_type == stencil.DynAccess and operand_to_inline in op.operands for op in consumer_apply.region.ops):
                continue

            return (producer_apply, access_op_to_inline_at, operand_to_inline)

    return None


def match_inlinable_old(consumer_apply: IOp) -> Optional[tuple[IOp, IOp, IResult]]:
    # Workaround until the registration and passing of values in native matchers is fixed
    if isinstance(consumer_apply, IResult):
        consumer_apply = consumer_apply.op
    # Explaining the matching:
    # - We match an apply op with operands and a region
    # - We check that one of the operands is another applyOp and remember it by `producer_apply`
    # - We check that inside the region is an AccessOp, that has the corresponding blockArg as operand and we remember it by `access_op_to_inline_at`
    # - we check that no empty stores are in the consumer apply + that no dynAccesses are in the consumer apply accessing the producer
    if consumer_apply.region is None or consumer_apply.region.block is None:
        return None
    if any((isinstance((operand), IResult) and (producer_apply := (operand_to_inline := operand).op).op_type == stencil.Apply) and \
                                any((consumer_apply.region.block.args[consumer_apply.operands.index(operand_to_inline)] in consumer_apply_op.operands) and (access_op_to_inline_at := consumer_apply_op).op_type == stencil.Access 
                                for consumer_apply_op in consumer_apply.region.ops)\
                                    and not any((consumer_apply_op.op_type == stencil.Store and len(consumer_apply.operands) == 0) | \
                                        (consumer_apply_op.op_type == stencil.DynAccess and operand_to_inline in consumer_apply_op.operands)\
                                    for consumer_apply_op in consumer_apply.region.ops)
                        for operand in consumer_apply.operands):
        return (producer_apply, access_op_to_inline_at, operand_to_inline)
    return None

def match_inlinable_native_escape(consumer_apply: IOp) -> Optional[tuple[IOp, IOp, IResult]]:
    # This is the matching code for the inlining rewrite defined in the rewriting dialect

    # Explaining the matching:
    # - We match an apply op with operands and a region
    # - We check that one of the operands is another applyOp and remember it by `producer_apply`
    # - We check that inside the region is an AccessOp, that has the corresponding blockArg as operand and we remember it by `access_op_to_inline_at`
    # - we check that no empty stores are in the consumer apply + that no dynAccesses are in the consumer apply accessing the producer
    if consumer_apply.region is None or consumer_apply.region.block is None:
        return None
    if any((isinstance((operand), IResult) and (producer_apply := (operand_to_inline := operand).op).op_type == stencil.Apply) and \
                                any((consumer_apply.region.block.args[consumer_apply.operands.index(operand_to_inline)] in consumer_apply_op.operands) and (access_op_to_inline_at := consumer_apply_op).op_type == stencil.Access 
                                for consumer_apply_op in consumer_apply.region.ops)\
                                    and not any((consumer_apply_op.op_type == stencil.Store and len(consumer_apply.operands) == 0) | \
                                        (consumer_apply_op.op_type == stencil.DynAccess and operand_to_inline in consumer_apply_op.operands)\
                                    for consumer_apply_op in consumer_apply.region.ops)
                        for operand in consumer_apply.operands):
        inlined_operand_index = producer_apply.results.index(operand_to_inline)
        return (producer_apply, access_op_to_inline_at, inlined_operand_index)
    return None

@dataclass(frozen=True)
class InlineApply(Strategy):
    """
    """
    def impl(self, op: IOp) -> RewriteResult:
        match consumer_apply := op:
            case IOp(op_type=stencil.Apply) if (matched_bits := match_inlinable(consumer_apply)):
                (producer_apply, access_op_to_inline_at, operand_to_inline) = matched_bits
                new_apply_operands: list[ISSAValue] = producer_apply.operands + consumer_apply.operands
                new_apply_block_args: list[IBlockArg] = producer_apply.region.block.args + consumer_apply.region.block.args

                assert access_op_to_inline_at is not None
                assert consumer_apply.region is not None
                assert consumer_apply.region is not None
                
                inlining_index = consumer_apply.region.ops.index(access_op_to_inline_at)
                
                new_apply_block = new_block(args=new_apply_block_args, 
                                            ops=consumer_apply.region.ops[0:inlining_index] + producer_apply.region.ops + consumer_apply.region.ops[inlining_index+1:], 
                                            env={},
                                            modify_op=self.handle_merging(producer_apply, consumer_apply, access_op_to_inline_at, operand_to_inline))

                new_apply = new_op(op_type=stencil.Apply, operands=new_apply_operands, 
                        result_types=consumer_apply.result_types, attributes=consumer_apply.attributes, 
                        regions=[IRegion([new_apply_block])])

                return success(new_apply)
            case _:
                return failure(self)

    def handle_merging(self, producer_apply: IOp, consumer_apply: IOp, access_op_to_inline_at: IOp, operand_to_inline: IResult) -> Callable[[IOp, Optional[dict[ISSAValue, ISSAValue]]], list[IOp]]:
        def merge(op_to_merge: IOp, env: Optional[dict[ISSAValue, ISSAValue]]) -> list[IOp]:
            assert producer_apply.region is not None
            if env is None:
                env = {}
            match op_to_merge:
                case IOp(op_type=stencil.StoreResult) if op_to_merge not in consumer_apply.region.ops:
                    assert op_to_merge.result is not None
                    env[op_to_merge.result] = env[op_to_merge.operands[0]] 
                    return []
                case IOp(op_type=stencil.Return) if op_to_merge not in consumer_apply.region.ops:
                    assert access_op_to_inline_at.result is not None
                    # Get the index of the operand to be inlined in the result of the producer apply and 
                    # map add the mapping of the access_op where we inline to the corresponding operand of stencil.result 
                    producer_result_index_to_inline = producer_apply.results.index(operand_to_inline)
                    env[access_op_to_inline_at.result] = env[op_to_merge.operands[producer_result_index_to_inline]]
                    return []
                case IOp(op_type=scf.If, regions=[IRegion() as then_region, IRegion() as else_region]) if op_to_merge not in consumer_apply.region.ops:
                    # We add the args to the env so we can just recurse here for the ops in the then and else region
                    for arg in producer_apply.region.block.args:
                        env[arg] = arg
                    then_region_ops: list[IOp] = []
                    for op in then_region.ops:
                        then_region_ops.extend(merge(op, env))
                    else_region_ops: list[IOp] = []
                    for op in else_region.ops:
                        else_region_ops.extend(merge(op, env))

                    return from_op(op_to_merge, regions=[IRegion([IBlock([], then_region_ops)]), IRegion([IBlock([], else_region_ops)])], env=env, 
                                    result_types=[res_type.elem if isinstance(res_type, stencil.ResultType) else res_type for res_type in op_to_merge.result_types])
                case IOp() if op_to_merge not in consumer_apply.region.ops:
                    # update the attributes `offset`, `lb`, `ub` for inlined ops
                    updated_attrs: dict[str, Attribute] = op_to_merge.attributes.copy()
                    if "offset" in updated_attrs:
                        updated_attrs["offset"] = updated_attrs["offset"] + access_op_to_inline_at.attributes["offset"]
                    if "lb" in updated_attrs:
                        updated_attrs["lb"] = updated_attrs["lb"] + access_op_to_inline_at.attributes["offset"]
                    if "ub" in updated_attrs:
                        updated_attrs["ub"] = updated_attrs["ub"] + access_op_to_inline_at.attributes["offset"]

                    return from_op(op_to_merge, attributes=updated_attrs, env=env)
                case _:
                    return from_op(op_to_merge, env=env)
        return merge

@dataclass(frozen=True)
class RerouteUse_decomp(Strategy):
    # In this rewrite we match two consumers that depend on the same producer
    fst_consumer: IOp
    producer: IOp

    # Matches a reroute target that has the producer as operand, exact same matching code as InlineApply
    # Used in matching of Strategy `RerouteOutputdependency`
    @dataclass(frozen=True)
    class MatchRerouteTargetDirectUse(Matcher):
        def apply(self, op: IOp) -> MatchResult:
            match consumer_apply := op:
                case IOp(op_type=stencil.Apply) if (matched_bits := match_inlinable(consumer_apply)):
                    (producer_apply, _, _) = matched_bits
                    if producer_apply == consumer_apply:
                        return match_failure(self)
                    return match_success([consumer_apply, producer_apply]) # cons_apply: 65, 66, 63, prod_apply: 67, 66, 63
                case _:
                    return match_failure(self)

    # Used when the producer is not directly used by the reroute target and thus can not be matched through it.
    # Used in matching of Strategy `RerouteInputdependency` 
    @dataclass(frozen=True)
    class MatchProducer(Matcher):
        def apply(self, op: IOp) -> MatchResult:
            match op:
                case IOp(op_type=stencil.Apply):
                    return match_success([op])
                case _:
                    return match_failure(self)

    # Matches a reroute target that has all operands of the producer as operand but not the producer itself
    # Used in matching of Strategy `RerouteInputdependency` 
    @dataclass(frozen=True)
    class MatchRerouteTargetParallelUse(Matcher):
        producer: IOp

        def apply(self, op: IOp) -> MatchResult:
            match op:
                case IOp(op_type=stencil.Apply, operands=[*operands]) \
                    if len(set(self.producer.operands).difference(operands)) == 0 and self.producer != op:
                    # Checked that all operands of the producer are also operands of the reroute target (i.e. all dependencies available)
                    return match_success([op, self.producer])
                case _:
                    return match_failure(self)


    # self.producer: apply with operands 67, 66, 63
    # self.fst_consumer: apply with operands: 65, 66, 63
    # snd_consumer: stencil.store with operand 65, 66, 63
    def impl(self, op: IOp) -> RewriteResult:
        match snd_consumer := op:
            # For a stencil.apply as snd_consumer the matching reuses the matching of InlineApply. For stencil.store as snd_consumer, we have match for the producer in the operands.
            case IOp(op_type=stencil.Apply | stencil.Store, operands=operands) if (((matched_bits := match_inlinable(snd_consumer)) and (producer_apply := matched_bits[0])) or \
                (any(isinstance((operand), IResult) and (producer_apply := operand.op).op_type == stencil.Apply for operand in operands))) \
                and snd_consumer != self.fst_consumer and self.producer == producer_apply:
                print("matched!")
                assert self.fst_consumer.region is not None
                assert self.fst_consumer.region.block is not None

                fst_consumer_new_operands: list[ISSAValue | IResult] = self.fst_consumer.operands + self.producer.results
                fst_consumer_new_block_args: list[IBlockArg] = self.fst_consumer.region.block.args + [IBlockArg(typ=result.typ, users=IList([]), block=None, index=len(self.fst_consumer.operands) + idx) for idx, result in enumerate(self.producer.results)]

                # Compute new bounds:
                new_fst_consumer_attr = self.fst_consumer.attributes.copy()
                new_fst_consumer_attr["lb"] = self.int_array_attr_element_wise(self.producer.attributes["lb"], self.fst_consumer.attributes["lb"], min)
                new_fst_consumer_attr["ub"] = self.int_array_attr_element_wise(self.producer.attributes["ub"], self.fst_consumer.attributes["ub"], max)
                
                fst_consumer_result_type = stencil.TempType.from_shape([new_fst_consumer_attr["ub"][idx] - new_fst_consumer_attr["lb"][idx] for idx in range(3)])
                
                # Build new fst consumer
                new_fst_consumer_block = from_block(self.fst_consumer.region.block, args=fst_consumer_new_block_args, modify_op=self.handle_merging(fst_consumer_new_operands, fst_consumer_new_block_args))
                new_fst_consumer = new_op(op_type=stencil.Apply, operands=fst_consumer_new_operands, 
                        result_types=[fst_consumer_result_type for _ in range(len(self.producer.results)+1)], attributes=new_fst_consumer_attr, 
                        regions=[IRegion([new_fst_consumer_block])])

                # Replace all uses of the producer with the new results of the fst_consumer
                new_snd_consumer_operands = snd_consumer.operands.copy()
                for idx in range(len(self.fst_consumer.results), len(new_fst_consumer[-1].results)):
                    if (old_operand := self.producer.results[idx- len(self.fst_consumer.results)]) in new_snd_consumer_operands:
                        new_snd_consumer_operands[new_snd_consumer_operands.index(old_operand)] = new_fst_consumer[-1].results[idx]

                result = success(new_fst_consumer, matched_op=self.fst_consumer)
                result += success(from_op(snd_consumer, operands=new_snd_consumer_operands), matched_op=snd_consumer)
                return result
            case _:
                return failure(self)
                
    @staticmethod
    def int_array_attr_element_wise(attr1: ArrayAttr, attr2: ArrayAttr, fun: Callable[[int, int], int]) -> ArrayAttr:
        return ArrayAttr.from_list([IntegerAttr.from_int_and_width(fun(int_attr1.value.data, int_attr2.value.data), 64) for int_attr1, int_attr2 in zip(attr1.data, attr2.data)])

    def handle_merging(self, fst_consumer_new_operands: list[ISSAValue], fst_consumer_new_block_args: list[IBlockArg]) -> Callable[[IOp, Optional[dict[ISSAValue, ISSAValue]]], list[IOp]]:
        def merge(op_to_merge: IOp, env: Optional[dict[ISSAValue, ISSAValue]]) -> list[IOp]:
            match op_to_merge:
                case IOp(op_type=stencil.Return):
                    result: list[IOp] = []
                    additional_return_vals : list[ISSAValue] = []
                    # Add a stencil.access and stencil.store_result for all new operands/block_args
                    for idx in range(len(self.fst_consumer.operands), len(fst_consumer_new_operands)):
                        result.extend(new_op(op_type=stencil.Access, operands=[fst_consumer_new_block_args[idx]], attributes={"offset" : ArrayAttr.from_list([IntegerAttr.from_int_and_width(0, 64) for _ in range(3)])}, result_types=[f64]))
                        assert result[-1].result is not None
                        result.extend(new_op(op_type=stencil.StoreResult, operands=[result[-1].result], result_types=[stencil.ResultType([f64])]))
                        additional_return_vals.append(result[-1].result)
                    result.extend(new_op(op_type=stencil.Return, operands=[operand for operand in op_to_merge.operands] + additional_return_vals))
                    return result
                case _:
                    return [op_to_merge]

        return merge

RerouteOutputDependency_decomp: Strategy = multiRoot(
            matchTopToBottom(RerouteUse_decomp.MatchRerouteTargetDirectUse()), lambda matched_consumer:
            topToBottom(RerouteUse_decomp(*matched_consumer)))

RerouteInputDependency_decomp: Strategy = multiRoot(
    matchSeq(
        matchTopToBottom(RerouteUse_decomp.MatchProducer()),
        lambda producer: matchTopToBottom(
            RerouteUse_decomp.MatchRerouteTargetParallelUse(*producer))),
    lambda matched_ops: topToBottom(RerouteUse_decomp(*matched_ops)))


StencilNormalForm: Strategy = everywhere(RemoveUnusedApplyOperands()) ^ everywhere(RemoveDuplicateApplyOperands()) ^ GarbageCollect()
InlineAll: Strategy = everywhere(RerouteOutputDependency_decomp) ^ everywhere(RerouteInputDependency_decomp) ^ everywhere(StencilNormalForm ^ InlineApply() ^ StencilNormalForm) ^ GarbageCollect()


##########################################################################
###################### Utils for rewriting dialect #######################
##########################################################################

def matchRerouteOutputDependency(fst_consumer: IOp) -> Optional[list[IOp, ISSAValue]]:
    """
    This is just temporary to enable testing the rerouting rewrites defined in the rewriting dialect without specifying the matching component.
    """
    producer_: Optional[IOp] = None
    fst_consumer_: Optional[IOp] = None
    snd_consumer_: Optional[IOp] = None

    @dataclass(frozen=True)
    class dummyStrat(Strategy):
        fst_consumer: IOp
        producer: IOp

        def impl(self, op: IOp) -> RewriteResult:
            match snd_consumer := op:
                # For a stencil.apply as snd_consumer the matching reuses the matching of InlineApply. For stencil.store as snd_consumer, we have match for the producer in the operands.
                case IOp(op_type=stencil.Apply | stencil.Store, operands=operands) if (((matched_bits := match_inlinable(snd_consumer)) and (producer_apply := matched_bits[0])) or \
                    (any(isinstance((operand), IResult) and (producer_apply := operand.op).op_type == stencil.Apply for operand in operands))) \
                    and snd_consumer != self.fst_consumer and self.producer == producer_apply and self.fst_consumer.op_type != func.FuncOp:
                    # currently we only get here when fst_consumer==snd_consumer
                    
                    nonlocal producer_
                    nonlocal fst_consumer_
                    nonlocal snd_consumer_

                    producer_ = producer_apply
                    fst_consumer_ = self.fst_consumer
                    snd_consumer_ = snd_consumer
                    return success(op)
                case _:
                    return failure(self)

    rr = multiRoot(matchTopToBottom(RerouteUse_decomp.MatchRerouteTargetDirectUse()), lambda matched_consumer:
            topToBottom(dummyStrat(*matched_consumer))).apply(fst_consumer)
    
    if rr.isSuccess():
        assert producer_ is not None
        assert snd_consumer_ is not None
        return [producer_, fst_consumer_, snd_consumer_]
    else:
        return None 