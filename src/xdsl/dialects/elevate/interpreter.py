from __future__ import annotations
from typing import Collection, Type
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.elevate import *
import xdsl.dialects.elevate.dialect as elevate_dialect
import xdsl.dialects.match.dialect as match
import xdsl.dialects.rewrite.dialect as rewrite
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf


@dataclass
class ElevateInterpreter():

    matching_env: dict[SSAValue, Attribute | ISSAValue] = field(default_factory=dict)
    native_matchers: dict[str, Callable[[IOp], Optional[Collection[IOp | IResult]]]] = field(default_factory=dict)
    native_strategies: dict[str, Type[Strategy]] = field(default_factory=dict)


    def register_native_matcher(self, matcher: Callable[[IOp], Optional[Collection[IOp | IResult]]], name: str):
        self.native_matchers[name] = matcher

    def register_native_rewriter(self):
        pass

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

        # get root_op
        root_op: Optional[match.RootOperationOp] = None
        root_index: int = 0
        for idx, nested_op in enumerate(strat_op.body.ops):
            if isinstance(nested_op, match.RootOperationOp):
                root_op = nested_op
                root_index = idx
                break
        assert root_op is not None

        if not self._match_for_op(op, root_op):
            # matching successful
            return failure(id())

        for idx in range(root_index+1, len(strat_op.body.ops)):
            if isinstance((cur_op := strat_op.body.ops[idx]), rewrite.SuccessOp):
                result = self.matching_env[cur_op.operands[0]]
                if isinstance(result, list):
                    return success(result)
                elif isinstance(result, IResult):
                    return success(result.op)
            if not self._interpret_rhs(cur_op): 
                return failure(id())

        # return success(op)
        return failure(id())



    def _match_for_op(self, op: IOp, op_constr: match.OperationOp) -> bool:
        # First check how many operands, attrs and other constraints are present:
        operand_constraints: list[SSAValue] = []
        result_constraints: list[SSAValue] = []
        attr_constraints: list[SSAValue] = []
        for idx, constraint in enumerate(op_constr.operands):
            assert isinstance(constraint, OpResult)
            match constraint.typ:
                case match.TypeType():
                    result_constraints.append(constraint)
                case match.ValueType():
                    operand_constraints.append(constraint)
                case _:
                    attr_constraints.append(constraint)
        
        if "name" in op_constr.attributes:
            if op_constr.attributes["name"].data != op.name:
                return False
            
        # General checks
        if len(operand_constraints) > 0 and len(op.operands) != len(operand_constraints):
            return False
        if len(result_constraints) > 0 and len(op.results) != len(result_constraints):
            return False
        # check result constraints:
        for idx, res_constraint in enumerate(result_constraints):
            if op.results[idx].typ != res_constraint.typ.type:
                return False
            
        # check attribute constraints:
        for idx, attr_constraint in enumerate(attr_constraints):
            assert isinstance((attr_op := attr_constraint.op), match.AttributeOp)
            # Check that the attribute with the given name exists
            if not attr_op.attributes["name"].data in op.attributes:
                return False
            # If a value is specified check that the attribute of op matches it
            if "value" in attr_op.attributes:
                if op.attributes[attr_op.attributes["name"].data] != attr_op.attributes["value"]:
                    return False
            self.matching_env[attr_constraint] = op.attributes[attr_op.attributes["name"].data]

        # check operand constraints:
        for idx, operand_constraint in enumerate(operand_constraints):
            match operand_constraint.op:
                case match.GetResultOp() as get_res_op:
                    if not isinstance((op.operands[idx]), IResult):
                        return False
                    if not get_res_op.idx.value.data == op.operands[idx].result_index: 
                        return False
                    if not self._match_for_op(
                            op=op.operands[idx].op,
                            op_constr=get_res_op.operands[0].op):
                        return False
                    self.matching_env[operand_constraint] = op.operands[idx]
                case match.OperationOp() | match.RootOperationOp() as op_match_op:
                    if not isinstance((op.operands[idx]), IResult):
                        return False
                    if not op_match_op.results.index(operand_constraint)-1 == op.operands[idx].result_index: 
                        return False
                    if not self._match_for_op(
                            op=op.operands[idx].op,
                            op_constr=op_match_op):
                        return False
                    self.matching_env[operand_constraint] = op.operands[idx]
                case match.OperandOp() as operand_op:
                    if len(operand_op.operands) == 1:
                        assert isinstance((type_constr := operand_op.operands[0].op), match.TypeOp)
                        if op.operands[idx].typ != type_constr.results[0].typ.type:
                            return False
                    self.matching_env[operand_constraint] = op.operands[idx]
                case _:
                    return False

        # record the match of this op in the matching environment so it can be referred to
        # in the rewriting part
        self.matching_env[op_constr.results[0]] = op
        if len(op_constr.results) > 1:
            for idx, op_constr_result in enumerate(op_constr.results[1:]):
                self.matching_env[op_constr_result] = op.results[idx]

        return True


    def _interpret_rhs(self, rhs_op: Operation) -> bool:
        match rhs_op:
            case arith.Addi():
                self.matching_env[rhs_op.results[0]] = self.matching_env[rhs_op.operands[0]] + self.matching_env[rhs_op.operands[1]]
                return True
            case rewrite.NewOp() | rewrite.FromOp():
                new_operands: list[ISSAValue] = []
                new_result_types: list[Attribute] = []
                attribute_operands: list[SSAValue] = []
                for operand in rhs_op.operands if isinstance(rhs_op, rewrite.NewOp) else rhs_op.operands[1:]:
                    match operand.typ:
                        case match.TypeType():
                            new_result_types.append(operand.typ.type)
                        case match.ValueType():
                            if isinstance(operand, OpResult) and isinstance(operand.op, match.GetResultOp):
                                new_operands.append(self.matching_env[operand.op.operands[0]])
                            else:
                                new_operands.append(self.matching_env[operand])
                        case _:
                            attribute_operands.append(operand)

                attributes: dict[str, Attribute] = {}
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
                        

                if isinstance(rhs_op, rewrite.NewOp):
                    if not "name" in rhs_op.attributes:
                        raise Exception("NewOp needs a name attribute")

                    # This is dangerous to do, but in this research context it is fine
                    op_type = eval(rhs_op.attributes["name"].data)
                    result = new_op(op_type, operands=new_operands, result_types=new_result_types, attributes=attributes)
                if isinstance(rhs_op, rewrite.FromOp):
                    result = from_op(self.matching_env[rhs_op.operands[0]], 
                                    operands=new_operands if len(new_operands) > 0 else None, 
                                    result_types=new_result_types if len(new_result_types) > 0 else None, 
                                    attributes=attributes if len(attributes) > 0 else None)

                self.matching_env[rhs_op.results[0]] = result
                # TODO: think about this properly!
                # Problem is when we just assign result[-1].results[idx] in the loop we lose the information
                # that the op was just created and has to be included in the return of the next `new_op`
                if len(rhs_op.results) > 1:
                    for idx, rhs_result in enumerate(rhs_op.results[1:]):
                        self.matching_env[rhs_result] = result

                return True
            case match.GetNestedOps() as get_payload_op:
                # return all ops in this region except for the terminator
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

                ops: list[IOp] = op.regions[region_idx].blocks[block_idx].ops

                if "exclude_terminator" in get_payload_op.attributes and get_payload_op.attributes["exclude_terminator"].value.data == 1:
                    # Ideally we would have a tag of whether an op is a terminator
                    if len(ops) > 0 and ops[-1].op_type == scf.Yield:
                        ops = ops[:-1]

                self.matching_env[get_payload_op.results[0]] = ops
                return True
            case match.NativeMatcherOp() as native_matcher_op:
                if not "matcher_name" in native_matcher_op.attributes:
                    raise Exception("NativeMatcherOp needs a matcher_name attribute")
                matcher_name: str = native_matcher_op.attributes["matcher_name"].data
                if not matcher_name in self.native_matchers:
                    raise Exception(f"NativeMatcherOp: Matcher {matcher_name} not registered")

                args = [self.matching_env[operand] for operand in native_matcher_op.operands]
                match_result : tuple = self.native_matchers[matcher_name](args[-1])
                if match_result is None:
                    return False

                if len(native_matcher_op.results) != len(match_result):
                    raise Exception(f"NativeMatcherOp: Matcher {matcher_name} returned {len(match_result)} results, but {len(native_matcher_op.results)} were expected")

                for idx, result in enumerate(match_result):
                    self.matching_env[native_matcher_op.results[idx]] = result
                
                return True
            case match.GetOperands() as get_operands_op:
                self.matching_env[get_operands_op.results[0]] = self.matching_env[get_operands_op.operands[0]].operands
                return True

            case _:
                return True


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