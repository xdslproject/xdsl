import json
from collections import defaultdict
from typing import Any, cast

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat
from xdsl.ir import BlockArgument, Operation
from xdsl.passes import ModulePass


class _IDGenerator:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.counter = 0

    def __call__(self):
        self.counter += 1
        return f"{self.prefix}{self.counter}"


def serialize_to_egraph(mod: builtin.ModuleOp):
    enode_to_id: defaultdict[Operation | BlockArgument, str] = defaultdict(
        _IDGenerator("enode_")
    )
    eclass_to_id: defaultdict[eqsat.AnyEClassOp, str] = defaultdict(
        _IDGenerator("eclass_")
    )
    nodes: dict[str, dict[str, str | list[str]]] = dict()
    for op in mod.walk(reverse=True):
        if isinstance(op, eqsat.AnyEClassOp):
            for operand in op.operands:
                if isinstance(operand, BlockArgument):
                    nodes[enode_to_id[operand]] = {
                        "op": f"arg {operand.index}",
                        "eclass": eclass_to_id[op],
                        "children": [],
                    }
            continue
        children: list[Any] = []
        eclass_id = None
        for res in op.results:
            for use in res.uses:
                if isinstance(use.operation, eqsat.AnyEClassOp):
                    assert len(op.results) == 1, (
                        "Only single result operations are supported"
                    )
                    assert res.has_one_use(), "Only single use operations are supported"
                    eclass_id = eclass_to_id[use.operation]
        if eclass_id is None:
            continue
        for operand in op.operands:
            if isinstance(operand.owner, eqsat.AnyEClassOp):
                firstoperand = operand.owner.operands[0]
                if isinstance(firstoperand.owner, Operation):
                    canonical_enode = firstoperand.owner
                else:
                    canonical_enode = cast(BlockArgument, firstoperand)
                children.append(enode_to_id[canonical_enode])
        if op.operands:
            name = op.name
        else:
            # If the operation has no operands, we get the full string representation as name for the node.
            # This is useful for operations such as `arith.constant 42`.
            name = str(op).split("=")[1].split(":")[0].strip()
        nodes[enode_to_id[op]] = {
            "op": name,
            "eclass": eclass_id,
            "children": children,
        }
    return nodes


class SerializeEGraph(ModulePass):
    name = "eqsat-serialize-egraph"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        print(json.dumps({"nodes": serialize_to_egraph(op)}))
