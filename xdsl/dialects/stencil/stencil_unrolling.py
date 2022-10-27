from __future__ import annotations
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
import xdsl.dialects.stencil.stencil as stencil


def get_shifted_attributes(attributes: dict[str, Attribute], offset: list[int]) -> Optional[dict[str, Attribute]]:
    """
    Takes in a dictionary of attributes and returns a new dictionary with the same keys but with the values shifted by the given offset. 
    Affected keys are: offset, lb, ub
    """


    if any([(do_offset_change := "offset" in attributes.keys()), (do_lb_change := "lb" in attributes.keys()), (do_ub_change := "ub" in attributes.keys())]):
        new_attributes = attributes.copy()
    
        def get_shifted_attr(attribute_name: str, offset: list[int]) -> ArrayAttr:
            old_op_attr: list[int] = [int_attr.value.data for int_attr in new_attributes[attribute_name].data]
            return ArrayAttr.from_list([IntegerAttr.from_int_and_width(offset[idx] + old_op_attr[idx], 64) for idx in range(3)])

        if do_offset_change:
            new_attributes["offset"] = get_shifted_attr("offset", offset)
        if do_lb_change:
            new_attributes["lb"] = get_shifted_attr("lb", offset)
        if do_ub_change:
            new_attributes["ub"] = get_shifted_attr("ub", offset)
        return new_attributes
    else:
        return None


@dataclass(frozen=True)
class UnrollApplyOp(Strategy):
    # In which dimension to unroll
    unroll_index: int = 1
    unroll_factor: int = 4

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=stencil.Apply, region=IRegion(block=IBlock(args=block_args, ops=ops))) if self.unroll_index <= 2:
                offset = [0,0,0]
                env: dict[ISSAValue, ISSAValue] = {}
                new_ops: List[IOp] = []
                new_results: List[ISSAValue] = []

                # Create new block args for the region of the apply op and add them to the env
                # The block field will be set by the IBlock when it is created
                new_block_args: List[IBlockArg] = [IBlockArg(typ=operand.typ, users=IList([]), block=None, index=idx) for idx, operand in enumerate(block_args)]
                for idx in range(len(new_block_args)):
                    env[block_args[idx]] = new_block_args[idx]

                for i in range(self.unroll_factor):
                    for nested_op in ops:
                        if nested_op.op_type == stencil.Return:
                            for result in nested_op.operands:
                                new_results.append(env[result])
                            continue

                        # Adjust attributes if necessary
                        if (new_attrs := get_shifted_attributes(nested_op.attributes, offset)) is not None:
                            new_ops.extend(from_op(nested_op, env=env, attributes=new_attrs))
                        else:
                            new_ops.extend(from_op(nested_op, env=env))
                    if i < self.unroll_factor-1:
                        offset[self.unroll_index] += 1

                # now add a new return op:
                new_ops.extend(new_op(stencil.Return, attributes={"unroll":ArrayAttr.from_list([IntegerAttr.from_int_and_width(offset[idx]+1, 64) for idx in range(3)])}, operands=new_results))
                
                # build new apply op:
                new_apply = from_op(op, regions=[IRegion(blocks=[IBlock(args=new_block_args, ops=new_ops)])])
                return success(new_apply)
            case _:
                return failure(self)