from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil.stencil as stencil
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *


def check_inlining_possible(producer: IOp, consumer: IOp, producer_result: ISSAValue) -> bool:
    if producer.region is None or consumer.region is None or consumer.region.block is None:
        return False

    # Check that there are no stencil.store ops in the producer.
    for nested_op in consumer.region.ops:
        if nested_op.op_type == stencil.Store and len(nested_op.operands) == 0:
            return False
            
    # Check that there is no stencil.dynAccess in the consumer accessing the producer 
    # (i.e. the blockArg associated with the producer)
    producer_idx = consumer.operands.index(producer_result)
    for nested_op in producer.region.ops:
        if nested_op.op_type == stencil.DynAccess and consumer.region.block.args[producer_idx] in nested_op.operands:
            return False
    return True

@dataclass(frozen=True)
class InlineProducer(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        # We match the consumer, rather than the producer.
        match op:
            case IOp(op_type=stencil.Apply,
                    operands=[ISSAValue(op=IOp(op_type=stencil.Apply) as producer_apply) as producer_result, *_] | [*_, ISSAValue(op=IOp(op_type=stencil.Apply) as producer_apply) as producer_result]) if check_inlining_possible(producer_apply, op, producer_result):
                env: dict[ISSAValue, ISSAValue] = {}
                
                # They just merge the args and canonicalize them away later. We will not do this!
                # i.e check before creation of the op, whether the args are actually used inside the region.

                # Create operands and blockArgs of the new ApplyOp:
                #   Merge operands of producer and consumer, no duplicate operands, and leave out the producer itself.
                #   Add a mapping of the old blockArgs to the new blockArgs.
                new_apply_operands: list[ISSAValue] = []
                new_apply_block_args: list[IBlockArg] = []
                for idx, operand in enumerate(producer_apply.operands + op.operands):
                    def add_mapping(index: int, block_arg: ISSAValue):
                        if index < len(producer_apply.operands):
                            env[producer_apply.region.block.args[index]] = block_arg
                        else: 
                            env[op.region.block.args[idx - len(producer_apply.operands)]] = block_arg

                    # Assuming no duplicated operands on the indicidual applys
                    # i.e. if there is a duplicate, the already added mapping is from the blockArg of the producer 
                    if operand in new_apply_operands:
                        block_arg = producer_apply.region.block.args[producer_apply.operands.index(operand)]
                        add_mapping(idx, env[block_arg])
                        continue
                    if operand in producer_apply.results:
                        # Will be inlined, not needed as operand
                        continue
                    new_apply_operands.append(operand)

                    # the block will set the block attribute here when it is created
                    new_apply_block_args.append(IBlockArg(typ=operand.typ, block=None, index=len(new_apply_block_args)))
                    add_mapping(idx, new_apply_block_args[-1])



                def get_ops_to_inline(old_ops: list[IOp], env: dict[ISSAValue, ISSAValue]) -> list[IOp]:
                    new_ops: list[IOp] = []

                    for old_op in old_ops:
                        if old_op.op_type == stencil.StoreResult:
                            # StoreResult denotes which value we have to use to replace the access op
                            assert access_op.result is not None
                            # In the inlined ops we move all references to the storeresult to its operand
                            env[old_op.result] = env[old_op.operands[0]]

                            # don't inline storeresult ops themselves
                            continue
                        elif old_op.op_type == stencil.Return:
                            # stencil.Return denotes which value we have to use to replace the access op
                            # The operand of the returnop has already been visited and inlined. We look it up in env
                            env[access_op.result] = env[old_op.operands[0]]

                            # don't inline the return op of the producer
                            continue
                        elif old_op.op_type == scf.If:
                            # rebuild the scf.If op
                            then_region_ops: list[IOp] = get_ops_to_inline(old_op.regions[0].ops, env=env)
                            else_region_ops: list[IOp] = get_ops_to_inline(old_op.regions[1].ops, env=env)
                            updated_scf_result_type = old_op.result_types[0].elem if isinstance(old_op.result_types[0], stencil.ResultType) else old_op.result_types[0]
                            new_ops.extend(from_op(old_op, regions=[IRegion([IBlock([], then_region_ops)]), IRegion([IBlock([], else_region_ops)])], env=env, result_types=[updated_scf_result_type]))
                            continue
                        # Ordinary inlining:
                        # Shift the producer op by the offset of the access op, i.e just adding the offsets
                        if "offset" in old_op.attributes.keys():
                            new_attributes = old_op.attributes.copy()

                            accessop_offset: list[int] = [int_attr.value.data for int_attr in access_op.attributes["offset"].data]
                            old_op_offset: list[int] = [int_attr.value.data for int_attr in new_attributes["offset"].data]

                            new_attributes["offset"] = ArrayAttr.from_list([IntegerAttr.from_int_and_width(accessop_offset[idx] + old_op_offset[idx], 64) for idx in range(3)])
                            new_ops.extend(from_op(old_op, env=env, attributes=new_attributes))
                        else:
                            new_ops.extend(from_op(old_op, env=env))                    
                    return new_ops

                # Create the region for the new ApplyOp
                new_apply_region_ops: list[IOp] = []
                for consumer_op in op.region.ops:
                    if (access_op := consumer_op).op_type == stencil.Access:
                        if op.operands[(accessed_block_arg_index := op.region.block.args.index(access_op.operands[0]))] in producer_apply.results:
                            # found an access op where I can do inlining!

                            # TODO: look in inlining cache for when the producer to be inlined has already been inlined
                            # We can do the inlining cache just with the env. So check here whether the access thingy is already in the env.

                            new_apply_region_ops.extend(get_ops_to_inline(producer_apply.region.ops, env=env))

                            # don't add the access op we do inlining for
                            continue

                    new_apply_region_ops.extend(from_op(consumer_op, env=env))

                # check whether all operands are actually used inside the region and remove duplicates
                new_apply = new_op(op_type=stencil.Apply, operands=new_apply_operands, 
                        result_types=op.result_types, attributes=op.attributes, 
                        regions=[IRegion([IBlock(new_apply_block_args, new_apply_region_ops)])])

                return success(new_apply)
            case _:
                return failure(self)

@dataclass(frozen=True)
class RerouteUse(Strategy):
    # In this rewrite we match two consumers that depend on the same producer
    fst_consumer: IOp
    producer: IOp

    @dataclass(frozen=True)
    class Match(Matcher):
        def apply(self, op: IOp) -> MatchResult:
            match op:
                case IOp(op_type=stencil.Apply,
                    operands=[ISSAValue(op=IOp(op_type=stencil.Apply) as producer) as producer_result, *_] | [*_, ISSAValue(op=IOp(op_type=stencil.Apply) as producer) as producer_result]) if check_inlining_possible(producer, op, producer_result):
                    print("found first match!")
                    return match_success([op, producer])
                case _:
                    return match_failure(self)

    def impl(self, op: IOp) -> RewriteResult:
        match snd_consumer := op:
            case IOp(op_type=stencil.Apply | stencil.Store,
                operands=[ISSAValue(op=IOp(op_type=stencil.Apply) as producer) as producer_result, *_] | [*_, ISSAValue(op=IOp(op_type=stencil.Apply) as producer) as producer_result]) if snd_consumer != self.fst_consumer and self.producer == producer and check_inlining_possible(producer, snd_consumer, producer_result):
                print("found second match!")
                # Do preprocessing of operands
                # Move operands referencing the producer from the snd_consumer to fst_consumer


                # Adjust bounds of fst_consumer
                producer_lb: list[int] = [int_attr.value.data for int_attr in producer.attributes["lb"].data]
                producer_ub: list[int] = [int_attr.value.data for int_attr in producer.attributes["ub"].data]
                fst_consumer_lb: list[int] = [int_attr.value.data for int_attr in self.fst_consumer.attributes["ub"].data]
                fst_consumer_ub: list[int] = [int_attr.value.data for int_attr in self.fst_consumer.attributes["ub"].data]

                new_fst_consumer_attr = self.fst_consumer.attributes.copy()
                new_fst_consumer_attr["lb"] = ArrayAttr.from_list([IntegerAttr.from_int_and_width(min(producer_lb[idx], fst_consumer_lb[idx]), 64) for idx in range(3)])
                new_fst_consumer_attr["ub"] = ArrayAttr.from_list([IntegerAttr.from_int_and_width(max(producer_ub[idx], fst_consumer_ub[idx]), 64) for idx in range(3)])
                
                print(f"adjusted bounds to: {new_fst_consumer_attr}")
                
                # Add stencil.storeResultOps for each of the moved operands + adjust return op <- this actually depends on being able to properly process multiple return values. 
                


                return success(op)
            case _:
                return failure(self)


# How is this called?
# multiRoot(
#         matchTopToBottom(RerouteUse.Match()), 
#         lambda matched_consumer: topToBottom(RerouteUse(*matched_consumer)))