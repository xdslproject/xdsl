from __future__ import annotations
import operator
from typing import Collection, Type
from xdsl.dialects.stencil.stencil_rewrites_decomposed import RerouteUse_decomp
from xdsl.immutable_utils import new_block
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *
from xdsl.elevate import *
import xdsl.dialects.elevate.dialect as elevate_dialect
from xdsl.dialects.IRUtils.dialect import ValueType, TypeType, OperationType, AnyType, AttributeType, RegionType, BlockType, RangeType, NativeHandleType
import xdsl.dialects.IRUtils.dialect as IRUtils
import xdsl.dialects.pdl.dialect as pdl
import xdsl.dialects.match.dialect as match
import xdsl.dialects.rewrite.dialect as rewrite
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil.stencil as stencil
import xdsl.dialects.onnx.dialect as onnx

@dataclass
class InterpResult:
    success: bool
    error_msg: str = ""

@dataclass
class ElevateInterpreter():

    matching_env: dict[SSAValue, Attribute | ISSAValue] = field(default_factory=dict)
    remapping_env: dict[SSAValue, Attribute | ISSAValue] = field(default_factory=dict)
    native_matchers: dict[str, Callable[[IOp], Optional[Collection[IOp | IResult]]]] = field(default_factory=dict)
    native_rewriters: dict[str, Callable[[IOp], Optional[Collection[IOp | IResult]]]] = field(default_factory=dict)
    native_strategies: dict[str, Type[Strategy]] = field(default_factory=dict)

    def register_native_matcher(self, matcher: Callable[[IOp], Optional[Collection[IOp | IResult]]], name: str):
        self.native_matchers[name] = matcher

    def register_native_rewriter(self, rewriter: Callable[[IOp], Optional[Collection[IOp | IResult]]], name: str):
        self.native_rewriters[name] = rewriter

    def register_native_strategy(self, strategy: Type[Strategy], name: str):
        self.native_strategies[name] = strategy

    def get_strategy(self,
                     strategy: elevate_dialect.ElevateOperation) -> dict[str, Strategy]:
        return self._dialect_to_strategies(strategy)

    def _dialect_to_strategies(self,
                             module: ModuleOp) -> dict[str, Strategy]:

        def get_strategy_for_region(region: Region) -> Strategy:
            assert len(region.ops) > 0
            strategy = get_strategy(region.ops[0])
            for idx in range(1, len(region.ops)):
                # TODO: assertion that all ops in the region are ElevateOps
                if isinstance(region.ops[idx], elevate_dialect.ReturnOp):
                    break
                strategy = seq(strategy, get_strategy(region.ops[idx]))
            return strategy

        def get_strategy(op: elevate_dialect.ElevateOperation) -> Strategy:
            # TODO: Move this to the match syntax when I don't need yapf anymore
            if isinstance(op, elevate_dialect.ComposeOp):
                return get_strategy_for_region(op.body)
            if isinstance(op, elevate_dialect.ApplyOp) and isinstance((compose_op := op.operands[0].op), elevate_dialect.ComposeOp):
                return get_strategy_for_region(compose_op.body)
            if isinstance(op, elevate_dialect.ApplyOp) and isinstance(op.operands[0].op, elevate_dialect.StrategyOp):
                return DynStrategy(op.strategy.op, self, op.args)
            if isinstance(op, elevate_dialect.ApplyOp) and isinstance((native_strat_op := op.operands[0].op), elevate_dialect.NativeStrategyOp):
                if not native_strat_op.strategy_name.data in self.native_strategies:
                    raise Exception(f"Native strategy: {native_strat_op.strategy_name.data} not registered!")
                return self.native_strategies[native_strat_op.strategy_name.data]()
            if isinstance(op, elevate_dialect.NativeStrategyOp):
                if not op.strategy_name.data in self.native_strategies:
                    raise Exception(f"Native strategy: {op.strategy_name.data} not registered!")
                return self.native_strategies[op.strategy_name.data]()
            if isinstance(op, elevate_dialect.StrategyOp):
                return DynStrategy(op, self, [])
            if len(op.regions) == 0:
                return op.get_strategy()()

            strategy_type: Type[Strategy] = op.__class__.get_strategy()
            operands_to_strat: list[Strategy] = [
                get_strategy_for_region(region) for region in op.regions
            ]

            return strategy_type(*operands_to_strat)


        strategies: dict[str, Strategy] = {}
        for nested_op in module.regions[0].blocks[0].ops:
            if isinstance(nested_op, elevate_dialect.ComposeOp) or isinstance(nested_op, elevate_dialect.StrategyOp) or isinstance(nested_op, elevate_dialect.NativeStrategyOp):
                if "strategy_name" in nested_op.attributes:
                    strategies[nested_op.attributes["strategy_name"].data] = get_strategy(nested_op)

        return strategies

    def interpret_strategy(self, strat_op: elevate_dialect.StrategyOp, op: IOp,
                           *args: ISSAValue) -> RewriteResult:
        # This is executed when a DynStrategy is actually applied

        # get match_and_replace op:
        replace_op: Optional[match.MatchAndReplace] = None
        for nested_op in reversed(strat_op.body.ops):
            if isinstance(nested_op, match.MatchAndReplace):
                replace_op = nested_op
                break
        assert replace_op is not None
        if replace_op is None:
            raise Exception("No match.match_and_replace found in strategy!")
        
        # check match_and_replace is well formed
        if len(replace_op.regions) != 1 or len(replace_op.regions[0].blocks) != 1:
            raise Exception("rewrite.replaceOp must have exactly one region with one block!")

        pattern : match.Pattern = replace_op.pattern.op

        # The last pdl.operation in the pattern is the root op
        root_op: Optional[pdl.OperationOp] = None
        capture_ops : List[match.Capture] = []
        for nested_op in pattern.body.ops:
            if isinstance(nested_op, pdl.OperationOp):
                root_op = nested_op
            elif isinstance(nested_op, match.Capture):
                capture_ops.append(nested_op)
        assert root_op is not None
        if root_op is None:
            raise Exception("No valid root operation found in pattern!")


        # matching part
        if not self._match_for_op(op, root_op).success:
            return failure(id())

        # registerd captured vals, ops, attrs
        captured_vals = [operand for capture_op in capture_ops for operand in capture_op.operands]


        if len(replace_op.body.blocks[0].args) == len(captured_vals):
            idx = 0
        elif len(replace_op.body.blocks[0].args) == len(captured_vals) + 1:
            idx = 1
            self.matching_env[replace_op.body.blocks[0].args[0]] = self.matching_env[root_op.results[0]]
        else:
            raise Exception("Number of arguments in replaceOp region must be equal to number of captured values or one more!")

        for captured_val in captured_vals:
            if not captured_val in self.matching_env:
                raise Exception(f"CaptureOp: {captured_val} not found in matching env!")
            self.matching_env[replace_op.body.blocks[0].args[idx]] = self.matching_env[captured_val]
            idx += 1

        # Matching successful

        # Interpret the region of replace op line by line until we hit a rewrite.return
        # for idx in range(root_index+1, len(strat_op.body.ops)):
        for rhs_op in replace_op.body.ops:
            if isinstance(rhs_op, rewrite.ReturnOp):
                if len(rhs_op.operands) > 1:
                    result = []
                    for operand in rhs_op.operands:
                        if operand in self.matching_env:
                            env_result = self.matching_env[operand]
                            result.extend(env_result if isinstance(env_result, list) else [env_result])
                elif len(rhs_op.operands) == 1:
                    result = self.matching_env[rhs_op.operands[0]]
                else:
                    raise Exception("rewrite.return must have at least one operand!")
                if isinstance(result, list):
                    return success(result)
                elif isinstance(result, IResult):
                    return success(result.op)
                elif isinstance(result, IOp):
                    return success([result])
                raise Exception("malformed rewrite.return operation")
            if not (interp_result := self._interpret_rhs(rhs_op)).success: 
                print(f"Failed to interpret with error msg: {interp_result}")
                return failure(id())

        # return success(op)
        return failure(id())


    def evaluate_operand_constraint(self,  op: IOp, idx: int, operand_constraint: SSAValue) -> InterpResult:
        match operand_constraint.op:
            case IRUtils.GetResults() as get_res_op:
                if not isinstance((op.operands[idx]), IResult):
                    return InterpResult(False, f"GetResultOp malformed")
                if not get_res_op.idx.value.data == op.operands[idx].result_index: 
                    return InterpResult(False, f"GetResultOp malformed")
                if not self._match_for_op(
                        op=op.operands[idx].op,
                        op_constr=get_res_op.operands[0].op):
                    return InterpResult(False, f"GetResultOp: Could not find match for op: {op.operands[idx].op.name}")
                self.matching_env[operand_constraint] = op.operands[idx]
                return InterpResult(True)
            case pdl.OperationOp() | rewrite.ReplaceOperationOp() as op_match_op:
                if not isinstance((op.operands[idx]), IResult):
                    return InterpResult(False, f"")
                if not op_match_op.results.index(operand_constraint)-1 == op.operands[idx].result_index: 
                    return InterpResult(False, f"")
                if not (result := self._match_for_op(
                        op=op.operands[idx].op,
                        op_constr=op_match_op)).success:
                    return result
                self.matching_env[operand_constraint] = op.operands[idx]
                return InterpResult(True)
            case pdl.OperandOp() as operand_op:
                if len(operand_op.operands) == 1:
                    assert isinstance((type_constr := operand_op.operands[0].op), pdl.TypeOp)
                    if "type" in type_constr.attributes:
                        enforced_type = type_constr.attributes["type"]
                        if op.operands[idx].typ == enforced_type:
                            self.matching_env[type_constr.results[0]] = op.operands[idx].typ
                        else:
                            return InterpResult(False, f"operand of op {op.name} has the wrong type")
                    else:
                        if type_constr.results[0] not in self.matching_env:
                            self.matching_env[type_constr.results[0]] = op.operands[idx].typ
                        elif self.matching_env[type_constr.results[0]] == op.operands[idx].typ:
                            pass
                        else:
                            return InterpResult(False, f"operand of op {op.name} has the wrong type")
                self.matching_env[operand_constraint] = op.operands[idx]
                return InterpResult(True)
            case match.AnyInRange() as any_in_range_op:
                # for now just assume that this is the range of results of an op
                range = self. any_in_range_op.range
                
                pass
            case _:
                return InterpResult(False, f"Could not resolve constraint")
    
    def _interpret_lhs_region_op(self, op_constr: pdl.OperationOp, region_owner: IOp) -> InterpResult:
        match op_constr:
            case IRUtils.GetIndex() as get_index_op:
                if get_index_op.value not in self.matching_env:
                    raise Exception(f"GetIndexOp: {get_index_op.value} not found in matching env!")
                value = self.matching_env[get_index_op.value]
                if isinstance(value, IResult):
                    self.matching_env[get_index_op.results[0]] = value.result_index
                elif isinstance(value, IBlockArg):
                    self.matching_env[get_index_op.results[0]] = value.index
                else:
                    raise Exception(f"GetIndexOp: {get_index_op.value} is not a result or block arg!")
                return InterpResult(True)
            case match.Equal() as equal_op:
                pass

            case _:
                return InterpResult(False, f"Could not resolve constraint {op_constr.name}")






    def _match_for_op(self, op: IOp, op_constr: pdl.OperationOp) -> InterpResult:
        # First check how many operands, attrs and other constraints are present:
        operand_constraints: list[SSAValue] = []
        result_constraints: list[SSAValue] = []
        attr_constraints: list[SSAValue] = []
        for idx, constraint in enumerate(op_constr.operands):
            assert isinstance(constraint, OpResult)
            match constraint.typ:
                case TypeType():
                    result_constraints.append(constraint)
                case ValueType():
                    operand_constraints.append(constraint)
                case _:
                    attr_constraints.append(constraint)
        
        if "name" in op_constr.attributes:
            if isinstance((name_attr := op_constr.attributes["name"].data), str) and name_attr != op.name:
                return InterpResult(False, f"name does not match: {name_attr} vs {op.name}")
            elif isinstance(name_attr, list) and op.name not in [attr.data for attr in name_attr]:
                return InterpResult(False, f"name does not match: {name_attr} vs {op.name}")

        # General checks
        if "operands_ordered" in op_constr.attributes and op_constr.attributes["operands_ordered"].data.value == 1:
            if len(operand_constraints) > 0 and len(op.operands) != len(operand_constraints):
                return InterpResult(False, f"wrong num of operands for op {op.name}")
        if len(result_constraints) > 0 and len(op.results) != len(result_constraints):
            return InterpResult(False, f"wrong num of results for op {op.name}")
        # check result constraints:
        for idx, res_constraint in enumerate(result_constraints):
            if "type" in res_constraint.op.attributes:
                enforced_type = res_constraint.op.attributes["type"]
                if op.results[idx].typ == enforced_type:
                    self.matching_env[res_constraint] = op.results[idx].typ
                    continue
                else:
                    return InterpResult(False, f"result with index {idx} of op {op.name} has the wrong type")
            else:
                if res_constraint not in self.matching_env:
                    self.matching_env[res_constraint] = op.results[idx].typ
                    continue
                elif self.matching_env[res_constraint] == op.results[idx].typ:
                    continue
                else:
                    return InterpResult(False, f"result with index {idx} of op {op.name} has the wrong type")

            
        # check attribute constraints:
        for idx, attr_constraint in enumerate(attr_constraints):
            assert isinstance((attr_op := attr_constraint.op), pdl.AttributeOp)
            # Check that the attribute with the given name exists
            if not (attr_name := attr_op.attributes["name"].data) in op.attributes:
                return InterpResult(False, f"Attr with name {attr_name} not found for op {op.name}")
            # If a value is specified check that the attribute of op matches it
            if "value" in attr_op.attributes:
                if op.attributes[(attr_name := attr_op.attributes["name"].data)] != attr_op.attributes["value"]:
                    return InterpResult(False, f"Attr with name {attr_name} has the wrong value for op {op.name}")
            self.matching_env[attr_constraint] = op.attributes[attr_op.attributes["name"].data]

        # Attr constraints expressed as attributes of pdl.operation:
        for attr_str in op_constr.attributes:
            if attr_str == "name" or attr_str == "operands_ordered":
                continue
            if not attr_str in op.attributes:
                return InterpResult(False, f"Attr with name {attr_str} not found for op {op.name}")
            if op.attributes[attr_str] != op_constr.attributes[attr_str]:
                return InterpResult(False, f"Attr with name {attr_str} has the wrong value for op {op.name}")

        # check operand constraints:
        for idx, operand_constraint in enumerate(operand_constraints):
            if ((interp_res := self.evaluate_operand_constraint(op, idx, operand_constraint)).success != True):
                return interp_res

        # check constraints about regions of this op
        for op_constr_region in op_constr.regions:

            if len(args := op_constr_region.blocks[0].args) == 1 and isinstance(args[0].typ, RangeType):
                self.matching_env[args[0]] = op.regions[0].blocks[0].args
            elif len(args) > 0:
                for idx, arg in enumerate(op_constr_region.blocks[0].args):
                    self.matching_env[arg] = op.regions[0].blocks[0].args[idx]
                

            for nested_constr in op_constr_region.ops:
                self._interpret_lhs_region_op(nested_constr, op)
        
        

        
        if isinstance(op_constr, rewrite.ReplaceOperationOp):
            env_binder_vals = op_constr.body.blocks[0].args
        elif isinstance(op_constr, pdl.OperationOp):
            env_binder_vals = op_constr.results
        else:
            raise Exception("malformed op constr")

        if len(env_binder_vals) == 0:
            raise Exception("No values (results or blockArgs) to bind the matched operation to.")

        # record the match of this op in the matching environment so it can be referred to
        # in the rewriting part
        self.matching_env[env_binder_vals[0]] = op

        # bind the results of the matched op the the results of the op constraint (either to a range or individually)
        if len(env_binder_vals) == 2 and isinstance(env_binder_vals[1].typ, RangeType):
            self.matching_env[env_binder_vals[1]] = op.results
        elif len(env_binder_vals) > 1:
            for idx, op_constr_result in enumerate(env_binder_vals[1:]):
                self.matching_env[op_constr_result] = op.results[idx]

        return InterpResult(True)

    # def _interpret_lhs(self, lhs_op: Operation) -> InterpResult:
    #     match lhs_op:
    #         case pdl.Operation() as pdl_operation:
                

    #             return InterpResult(True)
    #         case _:
    #             return InterpResult(False, f"Unknown lhs op {lhs_op.name} found in pattern")

    # def _interpret_pattern(self, pattern: match.Pattern) -> InterpResult:
    #     """
    #     Interprets a match.Pattern line by line and records matches in the matching environment.
    #     """
    #     for lhs_op in pattern.body.ops:
    #         if (interp_res := self._interpret_lhs(lhs_op)).success != True:
    #             return interp_res

    #     return InterpResult(True)

    def _interpret_rhs(self, rhs_op: Operation) -> InterpResult:
        match rhs_op:
            case arith.Addi():
                self.matching_env[rhs_op.results[0]] = self.matching_env[rhs_op.operands[0]] + self.matching_env[rhs_op.operands[1]]
                return InterpResult(True)
            case arith.Constant() as const_op:
                self.matching_env[rhs_op.results[0]] = const_op.attributes["value"].value.data
                return InterpResult(True)
            case IRUtils.NewOp() | IRUtils.FromOp():

                new_operands: list[ISSAValue] = []
                new_result_types: list[Attribute] = []
                attribute_operands: list[SSAValue] = []
                new_regions: list[IRegion] = []
                attributes: dict[str, Attribute] = {}
                for operand in rhs_op.operands if isinstance(rhs_op, IRUtils.NewOp) else rhs_op.operands[1:]:
                    match operand.typ:
                        case TypeType():
                            new_result_types.append(self.matching_env[operand])
                        case ValueType():
                            if isinstance(operand, OpResult) and isinstance(operand.op, IRUtils.GetResults):
                                new_operands.append(self.matching_env[operand.op.operands[0]])
                            else:
                                new_operands.append(self.matching_env[operand])
                        case RangeType() if isinstance(operand.typ.type, ValueType):
                            new_operands.extend(self.matching_env[operand])
                        case RangeType() if isinstance(operand.typ.type, TypeType):
                            new_result_types.extend(self.matching_env[operand])
                        case RangeType() if isinstance(operand.typ.type, RegionType):
                            new_regions.extend(self.matching_env[operand])
                        case RangeType() if isinstance(operand.typ.type, AttributeType):
                            attributes |= self.matching_env[operand]
                        case RegionType():
                            new_regions.append(self.matching_env[operand])
                        case _:
                            attribute_operands.append(operand)

                # Special casing for some specific ops so we don't have to e.g. provide a type that is easily inferrable
                if "name" in rhs_op.attributes and rhs_op.attributes["name"].data == "onnx.ONNXCastOp":
                    if len(rhs_op.operands) == 1 and "to" in rhs_op.attributes:
                        # i.e. no result type provided, so just infer it from the input and the "to" cast type
                        input_type = self.matching_env[rhs_op.operands[0]].typ
                        new_type = TensorType.from_type_and_list(rhs_op.attributes["to"], input_type.shape)
                        new_result_types.append(new_type)

                # Attributes are either specified as an operand when the attribute was computed in the rewrite
                # or as an attribute of the new op
                if "attribute_names" in rhs_op.attributes:
                    attribute_names: list[StringAttr] = rhs_op.attributes["attribute_names"].data
                    # assert len(attribute_names) == len(attribute_operands)

                    # for idx in range(len(attribute_operands)):
                    for attr_name in attribute_names:
                        if attr_name.data in rhs_op.attributes:
                            attributes[attr_name.data] = rhs_op.attributes[attr_name.data]
                            # attribute_operands.pop(0)
                        else:
                            attributes[attr_name.data] = self.matching_env[attribute_operands.pop(0)]
                    if len(attribute_operands) > 0:
                        raise Exception("Not all attributes were specified in \"attribute_names\"")
                        

                if isinstance(rhs_op, IRUtils.NewOp):
                    if not "name" in rhs_op.attributes:
                        raise Exception("NewOp needs a name attribute")

                    # This is dangerous to do, but in this research context it is fine
                    op_type = eval(rhs_op.attributes["name"].data)
                    result = new_op(op_type, operands=new_operands, result_types=new_result_types, attributes=attributes, regions=new_regions)
                if isinstance(rhs_op, IRUtils.FromOp):
                    result = from_op(self.matching_env[rhs_op.operands[0]], 
                                    operands=new_operands if len(new_operands) > 0 else None, 
                                    result_types=new_result_types if len(new_result_types) > 0 else None, 
                                    attributes=attributes if len(attributes) > 0 else None,
                                    regions=new_regions if len(new_regions) > 0 else None, env=self.remapping_env)

                self.matching_env[rhs_op.results[0]] = result
                # TODO: think about this properly!
                # Problem is when we just assign result[-1].results[idx] in the loop we lose the information
                # that the op was just created and has to be included in the return of the next `new_op`
                if len(rhs_op.results) > 1:
                    for idx, rhs_result in enumerate(rhs_op.results[1:]):
                        self.matching_env[rhs_result] = result

                # TODO: This is what we ideally would like to do here
                # if len(rhs_op.results) == 2 and isinstance(rhs_op.results[1].typ, RangeType):
                #     self.matching_env[rhs_op.results[1]] = result[-1].results
                # if len(rhs_op.results) > 1:
                #     for idx, rhs_result in enumerate(rhs_op.results[1:]):
                #         self.matching_env[rhs_result] = result[-1].results[idx]

                return InterpResult(True)
            case IRUtils.NewBlock() | IRUtils.FromBlock() as block_op:
                # Incomplete, only implemented for a basic case of new_block
                args = self.matching_env[block_op.operands[0]]
                ops = self.matching_env[block_op.operands[1]]
                custom_merger: Optional[Callable[[IOp, Optional[dict[ISSAValue, ISSAValue]]], List[IOp]]] = None
                if len(block_op.operands) > 2:
                    custom_merger = self.matching_env[block_op.operands[2]]

                result = new_block(args, ops, self.remapping_env, custom_merger)
                self.matching_env[block_op.results[0]] = result
                return InterpResult(True)
            case IRUtils.NewBlockArgs() as new_block_args_op:
                if not (isinstance(new_block_args_op.types.typ, RangeType) or isinstance(new_block_args_op.types.typ.type, ValueType)):
                    raise Exception("Malformed NewBlockArgsOp")
                types: list[Attribute] = self.matching_env[new_block_args_op.operands[0]]

                self.matching_env[new_block_args_op.results[0]] = [IBlockArg(typ=typ, block=None, index=idx) for idx, typ in enumerate(types)]
                
                return InterpResult(True)
            case IRUtils.RegionFromBlocks() as region_from_blocks_op:
                blocks: list[IBlock] = []
                for block_operand in region_from_blocks_op.operands:
                    if isinstance(block_operand.typ, BlockType):
                        blocks.append(self.matching_env[block_operand])
                    elif isinstance(block_operand.typ, RangeType):
                        blocks.extend(self.matching_env[block_operand])
                    else:
                        raise Exception("Unexpected operand type for RegionFromBlocksOp")
                self.matching_env[region_from_blocks_op.results[0]] = IRegion(blocks)
                return InterpResult(True)
            case IRUtils.Concat() as concat_op:
                if not all((concat_op.ranges[idx] in self.matching_env) for idx in range(len(concat_op.ranges))):
                    raise Exception("Malformed ConcatOp")
                new_list = []
                new_list += self.matching_env[concat_op.ranges[0]]
                for range_val in concat_op.ranges[1:]:
                    new_list += self.matching_env[range_val]

                # if isinstance(concat_op.results[0].typ.type, OperationType):
                #     tmp = []
                #     for elem in new_list:
                #         tmp.extend(from_op(elem, env=self.remapping_env))
                #     new_list = tmp

                self.matching_env[concat_op.output] = new_list
                return InterpResult(True)
            case IRUtils.GetElem() as get_elem_op:
                if not (get_elem_op.range in self.matching_env):
                    raise Exception("Malformed GetElemOp")
                index: int = 0
                if "index" in get_elem_op.attributes:
                    index = get_elem_op.attributes["index"].value.data
                elif len(get_elem_op.operands) == 2:
                    index = self.matching_env[get_elem_op.operands[1]]
                
                range_ = self.matching_env[get_elem_op.range]
                if index < 0:
                    index = len(range_) + index
                if index < 0 or index >= len(range_):
                    raise Exception("Index out of range")
                self.matching_env[get_elem_op.output] = range_[index]
                return InterpResult(True)
            case IRUtils.AttributeRange() as attribute_range_op:
                attribute_names: list[str] = []
                if "attribute_names" in attribute_range_op.attributes:
                    attribute_names = [name.data for name in attribute_range_op.attributes["attribute_names"].data]
                if isinstance(attribute_range_op.operands[0].typ, RangeType):
                    attributes = self.matching_env[attribute_range_op.operands[0]]
                else:
                    attributes: dict[str, Attribute] = {}
                for new_attr in attribute_range_op.operands:
                    if isinstance(new_attr.typ, RangeType):
                        attributes |= self.matching_env[new_attr]
                    elif isinstance(new_attr.typ, AttributeType):
                        if len(attribute_names) == 0:
                            raise Exception("AddAttributeOp: Attributenames underspecified")
                        attributes[attribute_names.pop(0)] = self.matching_env[new_attr]
                    else:
                        raise Exception("Unexpected operand type for AddAttributeOp")
                self.matching_env[attribute_range_op.output] = attributes
                return InterpResult(True)
            case IRUtils.GetNestedOps() as get_payload_op:
                # return all ops in this region
                op: IOp = self.matching_env[get_payload_op.operands[0]]
                if "region_idx" in get_payload_op.attributes:
                    region_idx: int = get_payload_op.attributes["region_idx"].value.data
                else:
                    region_idx: int = 0
                if "block_idx" in get_payload_op.attributes:
                    block_idx: int = get_payload_op.attributes["block_idx"].value.data
                else:
                    block_idx: int = 0
                if len(op.regions) <= region_idx:
                    raise Exception("get_payload_ops: Region index out of bounds")
                if len(op.regions[region_idx].blocks) <= block_idx:
                    raise Exception("get_payload_ops: Block index out of bounds")

                lb: int = 0
                ub: int = len(op.regions[region_idx].blocks[block_idx].ops)
                custom_lb = False
                if "custom_lb" in get_payload_op.attributes and get_payload_op.attributes["custom_lb"].value.data == 1:
                    custom_lb = True
                    lb = self.matching_env[get_payload_op.operands[1]]
                if "custom_ub" in get_payload_op.attributes and get_payload_op.attributes["custom_ub"].value.data == 1:
                    if custom_lb:
                        ub = self.matching_env[get_payload_op.operands[2]]
                    else:
                        ub = self.matching_env[get_payload_op.operands[1]]


                ops: list[IOp] = op.regions[region_idx].blocks[block_idx].ops[lb:ub]

                if "exclude_terminator" in get_payload_op.attributes and get_payload_op.attributes["exclude_terminator"].value.data == 1:
                    # Ideally we would have a tag of whether an op is a terminator
                    if len(ops) > 0 and (ops[-1].op_type == scf.Yield or ops[-1].op_type == stencil.Return):
                        ops = ops[:-1]

                self.matching_env[get_payload_op.results[0]] = ops
                return InterpResult(True)
            case IRUtils.NativeMatcher() as native_matcher_op:
                if not "matcher_name" in native_matcher_op.attributes:
                    raise Exception("NativeMatcherOp needs a matcher_name attribute")
                matcher_name: str = native_matcher_op.attributes["matcher_name"].data
                if not matcher_name in self.native_matchers:
                    raise Exception(f"NativeMatcherOp: Matcher {matcher_name} not registered")

                args = [self.matching_env[operand] for operand in native_matcher_op.operands]
                match_result : tuple = self.native_matchers[matcher_name](args[-1])
                if match_result is None:
                    return InterpResult(False, f"Native matcher {matcher_name} did not match")

                if len(native_matcher_op.results) != len(match_result):
                    raise Exception(f"NativeMatcherOp: Matcher {matcher_name} returned {len(match_result)} results, but {len(native_matcher_op.results)} were expected")

                for idx, result in enumerate(match_result):
                    self.matching_env[native_matcher_op.results[idx]] = result
                
                return InterpResult(True)
            case IRUtils.ApplyNativeRewrite() as native_rewriter_op:
                if not "rewriter_name" in native_rewriter_op.attributes:
                    raise Exception("NativeRewriterOp needs a rewriter_name attribute")
                rewriter_name: str = native_rewriter_op.attributes["rewriter_name"].data
                if not rewriter_name in self.native_rewriters:
                    raise Exception(f"NativeRewriterOp: Rewriter {rewriter_name} not registered")

                if not all((operand in self.matching_env) for operand in native_rewriter_op.operands):
                    raise Exception("Malformed NativeRewriterOp")
                args = [self.matching_env[operand] for operand in native_rewriter_op.operands]
                self.matching_env[native_rewriter_op.results[0]] = self.native_rewriters[rewriter_name](*args)
                return InterpResult(True)
            case IRUtils.GetOp() as get_op_op:
                if get_op_op.operands[0] not in self.matching_env or not isinstance((result := self.matching_env[get_op_op.operands[0]]), IResult):
                    raise Exception("Malformed GetOpOp")
                self.matching_env[get_op_op.results[0]] = result.op
                return InterpResult(True)
            case IRUtils.GetResults() as get_results_op:
                if "index" in get_results_op.attributes and isinstance(get_results_op.results[0].typ, ValueType):
                    idx: int = get_results_op.attributes["index"].value.data
                    if idx < 0 or idx >= len(self.matching_env[get_results_op.operands[0]].results):
                        raise Exception("GetResultsOp: index out of bounds")
                    self.matching_env[get_results_op.results[0]] = self.matching_env[get_results_op.operands[0]].results[idx]
                elif isinstance(get_results_op.results[0].typ, RangeType) and isinstance(get_results_op.results[0].typ.type, ValueType):
                    self.matching_env[get_results_op.results[0]] = self.matching_env[get_results_op.operands[0]].results
                else:
                    raise Exception("Malformed GetResultsOp")
                
                return InterpResult(True)
            case IRUtils.GetOperands() as get_operands_op:
                self.matching_env[get_operands_op.results[0]] = self.matching_env[get_operands_op.operands[0]].operands
                return InterpResult(True)
            case IRUtils.GetOperand() as get_operand_op:
                if not (get_operand_op.input in self.matching_env):
                    raise Exception("Malformed GetOperandOp")
                index: int = 0
                if "index" in get_operand_op.attributes:
                    index = get_operand_op.attributes["index"].value.data
                elif len(get_operand_op.operands) == 2:
                    index = self.matching_env[get_operand_op.operands[1]]
                op = self.matching_env[get_operand_op.operands[0]]
                if index < 0:
                    index = len(op.operands) + index
                if index < 0 or index >= len(op.operands):
                    raise Exception("GetOperandOp: index out of bounds")
                self.matching_env[get_operand_op.results[0]] = op.operands[index]
                return InterpResult(True)
            case IRUtils.GetAttributes() as get_attributes_op:
                self.matching_env[get_attributes_op.results[0]] = self.matching_env[get_attributes_op.operands[0]].attributes
                return InterpResult(True)
            case IRUtils.GetAttribute() as get_attributes_op:
                if not "attr_name" in get_attributes_op.attributes:
                    raise Exception("GetAttributeOp needs an attr_name attribute")
                attr_name: str = get_attributes_op.attributes["attr_name"].data
                if not attr_name in self.matching_env[get_attributes_op.operands[0]].attributes:
                    raise Exception(f"GetAttributeOp: Attribute {attr_name} not found")
                self.matching_env[get_attributes_op.results[0]] = self.matching_env[get_attributes_op.operands[0]].attributes[attr_name]
                return InterpResult(True)
            case IRUtils.HasAttribute() as has_attribute_op:
                op = self.matching_env[has_attribute_op.operands[0]]
                if not "attr_name" in has_attribute_op.attributes:
                    raise Exception("HasAttribute needs an attr_name attribute")
                attr_name: str = has_attribute_op.attributes["attr_name"].data
                self.matching_env[has_attribute_op.results[0]] = (attr_name in op.attributes)
                return InterpResult(True)
            case IRUtils.ArrayAttrElementWise() as array_attr_element_wise_op:
                if not "op" in array_attr_element_wise_op.attributes:
                    raise Exception("ArrayAttrElementWiseOp needs an op attribute")
                if not (isinstance((array0 := self.matching_env[array_attr_element_wise_op.array0]), ArrayAttr) and isinstance((array1 := self.matching_env[array_attr_element_wise_op.array1]), ArrayAttr)):
                    raise Exception("ArrayAttrElementWiseOp: Operands must be ArrayAttr")
                match array_attr_element_wise_op.attributes["op"].data:
                    case "add":
                        self.matching_env[array_attr_element_wise_op.results[0]] = RerouteUse_decomp.int_array_attr_element_wise(array0, array1, operator.__add__)
                    case "sub":
                        self.matching_env[array_attr_element_wise_op.results[0]] = RerouteUse_decomp.int_array_attr_element_wise(array0, array1, operator.__sub__)
                    case "min":
                        self.matching_env[array_attr_element_wise_op.results[0]] = RerouteUse_decomp.int_array_attr_element_wise(array0, array1, min)
                    case "max":
                        self.matching_env[array_attr_element_wise_op.results[0]] = RerouteUse_decomp.int_array_attr_element_wise(array0, array1, max)
                    case _:
                        raise Exception("ArrayAttrElementWiseOp: Unknown op")
                return InterpResult(True)
            case IRUtils.GetBlockArgs() as get_blockargs_op:
                region_idx = 0
                block_idx = 0
                if "region_idx" in get_blockargs_op.attributes:
                    region_idx = get_blockargs_op.attributes["region_idx"].value.data
                if "block_idx" in get_blockargs_op.attributes:
                    block_idx = get_blockargs_op.attributes["block_idx"].value.data
                self.matching_env[get_blockargs_op.results[0]] = self.matching_env[get_blockargs_op.operands[0]].regions[region_idx].blocks[block_idx].args
                return InterpResult(True)
            case IRUtils.GetType() as get_type_op:
                if isinstance(get_type_op.operands[0].typ, RangeType) and isinstance(get_type_op.operands[0].typ.type, ValueType):
                    self.matching_env[get_type_op.results[0]] = [value.typ for value in self.matching_env[get_type_op.operands[0]]]
                elif isinstance(get_type_op.operands[0].typ, ValueType):
                    env_operand = self.matching_env[get_type_op.operands[0]]
                    # This is a workaround for the issue related to the mapping to results on the rhs
                    if isinstance(env_operand, list):
                        env_operand = env_operand[0].results[0]
                    self.matching_env[get_type_op.results[0]] = env_operand.typ
                else:
                    raise Exception("Malformed GetTypeOp")
                return InterpResult(True)
            case IRUtils.ConstructType() as construct_type_op:
                if "name" not in construct_type_op.attributes:
                    raise Exception("ConstructTypeOp needs a name attribute")
                # again, very unsafe
                #type = eval(rhs_op.attributes["name"].data)(*[self.matching_env[operand] for operand in construct_type_op.operands])
                # This should actually work, but not sure whether something is wrongly defined in TempType
                
                if rhs_op.attributes["name"].data == "stencil.TempType":
                    type = stencil.TempType.from_shape(self.matching_env[construct_type_op.operands[0]])
                    self.matching_env[construct_type_op.results[0]] = type
                else:
                    raise Exception("ConstructTypeOp: Type not supported")
                return InterpResult(True)
            case IRUtils.GetIndexOfOpInRange() as get_index_op:
                if not all((get_index_op.operands[idx] in self.matching_env) for idx in range(len(get_index_op.operands))):
                    raise Exception("Malformed GetIndexOfOpInRange")
                self.matching_env[get_index_op.results[0]] = self.matching_env[get_index_op.operands[1]].index(self.matching_env[get_index_op.operands[0]])
                return InterpResult(True)
            case IRUtils.RemoveElement() as remove_element_op:
                if not (remove_element_op.range in self.matching_env):
                    raise Exception("Malformed RemoveElementOp")
                index: int = 0
                if "index" in remove_element_op.attributes:
                    index = remove_element_op.attributes["index"].value.data
                elif len(remove_element_op.operands) == 2:
                    index = self.matching_env[remove_element_op.operands[1]]
                
                range_ = self.matching_env[remove_element_op.range]
                if index < 0:
                    index = len(range_) + index
                if index < 0 or index >= len(range_):
                    raise Exception("Index out of range")
                range_.pop(index)
                self.matching_env[remove_element_op.range] = range_
                return InterpResult(True)
            case IRUtils.ForEach() as for_each_op:
                # self.matching_env[for_each_op.results[0]] = self.matching_env[for_each_op.operands[0]]
                list_ = self.matching_env[for_each_op.operands[0]]
                result_list = []
                if not isinstance(list_, list):
                    raise Exception("ForEachOp: First operand must be a range")
                if len(for_each_op.regions) != 1 or len(for_each_op.regions[0].blocks) != 1 or len(for_each_op.regions[0].blocks[0].args) != 1:
                    raise Exception("ForEachOp: Malformed region")
                for elem in list_:
                    self.matching_env[for_each_op.regions[0].blocks[0].args[0]] = elem

                    for nested_op in for_each_op.regions[0].blocks[0].ops:
                        if not isinstance(nested_op, IRUtils.Yield):
                            self._interpret_rhs(nested_op)
                        else:
                            result = self.matching_env[nested_op.operands[0]]
                            if isinstance(result, list):
                                # self.matching_env[nested_op.operands[0]] = result[0]
                                result_list.extend(result)
                            else:
                                # self.matching_env[nested_op.operands[0]] = result
                                result_list.append(result)
                            # result_list.append(elem)
                            continue
                self.matching_env[for_each_op.results[0]] = result_list
                return InterpResult(True)
            case IRUtils.If() as if_op:
                condition = self.matching_env[if_op.operands[0]]
                ops_to_interpret = if_op.regions[0].blocks[0].ops
                if len(if_op.regions) == 2 and not condition:
                    ops_to_interpret = if_op.regions[1].blocks[0].ops

                for nested_op in ops_to_interpret:
                    if isinstance(nested_op, IRUtils.Yield):
                        self.matching_env[if_op.results[0]] = self.matching_env[nested_op.operands[0]]
                        return InterpResult(True)
                    else:
                        self._interpret_rhs(nested_op)
                
                if len(if_op.results) > 0:
                    raise Exception("IfOp: If Op returns a result but no yield was encountered")
                return InterpResult(True)
            case IRUtils.ReplaceUses() as replace_uses_op:
                old_use = self.matching_env[replace_uses_op.old_use]
                new_use = self.matching_env[replace_uses_op.new_use]

                self.remapping_env[old_use] = new_use
                return InterpResult(True)
            case IRUtils.ConcatTensors() as concat_tensors_op:
                if not all((operand in self.matching_env) for operand in concat_tensors_op.operands):
                    raise Exception("Malformed ConcatTensorsOp")
                values: list[list[float]] = []
                for operand in concat_tensors_op.operands:
                    tensor = self.matching_env[operand]
                    # for now assume that the tensor stems from an op which has a `value` attribute
                    value_attr_list = tensor.op.attributes["value"].data.data
                    value_list = [float_attr.value.data for float_attr in value_attr_list]
                    values.append(value_list)
                

                if all((len(value) == 1 and value[0] == 1.0) for value in values) and all(len(value) == len(values[0]) for value in values):
                    new_shape = tensor.typ.shape.get_as_data_array()
                    new_shape[-1] = new_shape[-1] * len(values)

                    type = TensorType.from_type_and_list(tensor.typ.element_type, new_shape)

                    # new_shape = ArrayAttr(IntegerAttr.from_index_int_value(new_shape[1].value.data * len(values))
                    self.matching_env[concat_tensors_op.results[0]] = DenseIntOrFPElementsAttr.from_list(
                        type, [1.0])
                    if len(concat_tensors_op.results) == 2 and isinstance(concat_tensors_op.results[1].typ, TypeType):
                        self.matching_env[concat_tensors_op.results[1]] = type


                else:
                    raise Exception("ConcatTensorsOp: Not implemented")



                return InterpResult(True)

            case _:
                raise Exception(f"Op: {rhs_op.name} unsupported by interpreter")


@dataclass(frozen=True)
class DynStrategy(Strategy):
    strat_op: elevate_dialect.StrategyOp
    interpreter: ElevateInterpreter
    args: list[ISSAValue]

    def impl(self, op: IOp) -> RewriteResult:
        return self.interpreter.interpret_strategy(  # type: ignore
            self.strat_op, op, *self.args)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'