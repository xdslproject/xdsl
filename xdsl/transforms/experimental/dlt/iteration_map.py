from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern, attr_type_rewrite_pattern, \
    op_type_rewrite_pattern


class DLTIterateConsistencyError(Exception):
    pass


class IterationMap:
    def __init__(self, iteration_ops: dict[StringAttr, dlt.IterateOp]):
        self.iteration_ops = iteration_ops
        for name, iter_op in iteration_ops.items():
            if "identification" not in iter_op.attributes:
                raise ValueError("Iteration ops must have identification attribute")
            if iter_op.attributes["identification"] != name:
                raise ValueError(f"Iteration op identification attribute is not set to {name}")
            if "iter_order" not in iter_op.attributes:
                raise ValueError("Iteration ops must have iter_order attribute")

    def add(self, iteration_op: dlt.IterateOp):
        assert iteration_op.identification != StringAttr("")
        self.iteration_ops[iteration_op.identification] = iteration_op

    def get_map(self) -> dict[StringAttr, dlt.IterationOrder]:
        return {name: op.order for name, op in self.iteration_ops.items()}

    def check_consistency(self, map: dict[StringAttr, dlt.IterationOrder] = None):
        if map is None:
            map = self.get_map()

        if set(map.keys()) & set(self.iteration_ops.keys()) != set(self.iteration_ops.keys()):
            raise ValueError("map does not have iteration order for all iterate ops in IterationMap")

        for name, order in map.items():
            op = self.iteration_ops[name]
            op.verify_order(order, e=DLTIterateConsistencyError)
        return True

    def is_consistent(self, new_types: dict[StringAttr, dlt.PtrType] = None):
        try:
            check = self.check_consistency(new_types)
            return check
        except DLTIterateConsistencyError as e:
            return False

    def use_orders(self, new_orders: dict[StringAttr, dlt.IterationOrder]):
        if any(i not in new_orders for i in self.iteration_ops):
            raise ValueError(
                "new_orders does not have a type for all the iterate ops in the map"
            )
        type_rewriter = PatternRewriteWalker(IterateOpOrderRewriter(new_orders))
        return type_rewriter


class IterateOpOrderRewriter(RewritePattern):

    def __init__(self, ptr_map: dict[StringAttr, dlt.IterationOrder]):
        self.ptr_map = ptr_map
        super().__init__()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dlt.IterateOp, rewriter: PatternRewriter, /):
        if op.has_identity and op.identification in self.ptr_map:
            new_order = self.ptr_map[op.identification]
            if op.order != new_order:
                op.order = new_order
                rewriter.handle_operation_modification(op)