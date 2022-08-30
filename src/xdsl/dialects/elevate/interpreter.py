from __future__ import annotations
from typing import Type
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.elevate import *
import xdsl.dialects.elevate.dialect as elevate_dialect
import xdsl.dialects.match.dialect as match
import xdsl.dialects.rewrite.dialect as rewrite
import xdsl.dialects.arith as arith


@dataclass
class ElevateInterpreter():

    matching_env: dict[SSAValue, Attribute | ISSAValue] = field(default_factory=dict)

    def register_native_matcher(self):
        pass

    def register_native_rewriter(self):
        pass

    def get_strategy(self,
                     strategy: elevate_dialect.ElevateOperation) -> Strategy:
        return self._dialect_to_strategy(strategy)

    def _dialect_to_strategy(self,
                             strategy: elevate_dialect.StrategyOp) -> Strategy:

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
            if isinstance(op, ModuleOp):
                for nested_op in op.regions[0].blocks[0].ops:
                    if isinstance(nested_op, elevate_dialect.ComposeOp):
                        return get_strategy(nested_op)
                raise Exception("No elevate.compose Operation found!")
            if isinstance(op, elevate_dialect.ComposeOp):
                return get_strategy_for_region(op.body)
            if isinstance(op, elevate_dialect.ApplyOp):
                return DynStrategy(op.strategy.op, self, op.args)
            if len(op.regions) == 0:
                return op.get_strategy()()

            strategy_type: Type[Strategy] = op.__class__.get_strategy()
            operands_to_strat: list[Strategy] = [
                get_strategy_for_region(region) for region in op.regions
            ]

            return strategy_type(*operands_to_strat)

        return get_strategy(strategy)

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
                return success(self.matching_env[cur_op.operands[0]])
            self._interpret_rhs(cur_op)

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
        
        # TODO: also check for name!
        if "name" in op_constr.attributes:
            if op_constr.attributes["name"].data != op.name:
                return False
            
        # General checks
        if len(op.operands) != len(operand_constraints) or len(op.results) != len(result_constraints):
            return False


        # check result constraints:
        for idx, res_constraint in enumerate(result_constraints):
            if op.results[idx].typ != res_constraint.typ.type:
                return False
            
        # check attribute constraints:
        for idx, attr_constraint in enumerate(attr_constraints):
            assert isinstance((attr_op := attr_constraint.op), match.AttributeOp)
            if not attr_op.attributes["name"].data in op.attributes:
                return False
            self.matching_env[attr_constraint] = op.attributes[attr_op.attributes["name"].data]

        # check operand constraints:
        for idx, operand_constraint in enumerate(operand_constraints):
            assert isinstance((get_res_op := operand_constraint.op), match.GetResultOp)
            if not isinstance((op.operands[idx]), IResult):
                return False
            if not get_res_op.idx.value.data == op.operands[
                    idx].result_index:  # type: ignore
                return False
            if not self._match_for_op(
                    op=op.operands[idx].op,
                    op_constr=get_res_op.operands[0].op):  # type: ignore
                return False
            self.matching_env[operand_constraint] = op.operands[idx]

        return True

        # Then recurse into the values that are input to this operation


    def _interpret_rhs(self, rhs_op: Operation):
        match rhs_op:
            case arith.Addi():
                self.matching_env[rhs_op.results[0]] = self.matching_env[rhs_op.operands[0]] + self.matching_env[rhs_op.operands[1]]
            case rewrite.NewOp() as new_op_op:
                # TODO: op_type should come from the attribute of new_op_op

                new_operands: list[ISSAValue] = []
                new_result_types: list[Attribute] = []
                attribute_operands: list[SSAValue] = []
                for operand in new_op_op.operands:
                    match operand.typ:
                        case match.TypeType():
                            new_result_types.append(operand.typ.type)
                        case match.ValueType():
                            new_operands.append(self.matching_env[operand])
                        case _:
                            attribute_operands.append(operand)

                attributes: dict[str, Attribute] = {}
                if "attribute_names" in new_op_op.attributes:
                    attribute_names: list[StringAttr] = new_op_op.attributes["attribute_names"].data
                    assert len(attribute_names) == len(attribute_operands)
                    for idx in range(len(attribute_operands)):
                        attributes[attribute_names[idx].data] = self.matching_env[attribute_operands[idx]]

                result = new_op(arith.Constant, operands=new_operands, result_types=new_result_types, attributes=attributes)
                self.matching_env[rhs_op.results[0]] = result
            case _:
                return


@dataclass(frozen=True)
class DynStrategy(Strategy):
    strat_op: elevate_dialect.StrategyOp
    interpreter: ElevateInterpreter
    args: list[ISSAValue]

    def impl(self, op: IOp) -> RewriteResult:
        print("here we would do the interpretation")
        return self.interpreter.interpret_strategy(  # type: ignore
            self.strat_op, op, *self.args)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}()'