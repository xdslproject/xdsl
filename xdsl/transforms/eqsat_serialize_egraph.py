import json

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat, func
from xdsl.ir import BlockArgument, Operation
from xdsl.passes import ModulePass


def build_egraph_dict(f_op: func.FuncOp):
    assert len(f_op.regions) == 1
    body = f_op.regions[0]
    assert body.first_block == body.last_block, (
        "Only single block functions are supported"
    )
    body = body.first_block

    i_eclass = 0
    i_enode = 0
    eclass_to_id: dict[eqsat.EClassOp, int] = {}
    op_to_id = {}
    nodes: list[dict[str, str | list[str]]] = []

    if body:
        for op in body.ops:
            if isinstance(op, eqsat.EClassOp):
                eclass_to_id[op] = (
                    i_enode  # pick the first e-node as the representative
                )
                for operand in op.operands:
                    node = operand.owner
                    children: list[str] = []
                    if isinstance(node, Operation):
                        op_to_id[node] = i_enode
                        if node.operands:
                            name = node.name
                        else:
                            name = str(node).split("=")[1].split(":")[0].strip()
                        for child_op in node.operands:
                            assert isinstance(child_op.owner, eqsat.EClassOp)
                            child = eclass_to_id.get(child_op.owner)
                            assert child is not None, "Child eclass not found"
                            children.append(f"enode_{child}")
                    else:
                        assert isinstance(operand, BlockArgument)
                        op_to_id[operand] = i_enode
                        name = f"arg {operand.index}"

                    nodes.append(
                        {
                            "op": name,
                            "eclass": f"eclass_{i_eclass}",
                            "children": children,
                        }
                    )
                    i_enode += 1
                i_eclass += 1
    return nodes


def serialize_egraph_to_json(f_op: func.FuncOp):
    nodes = build_egraph_dict(f_op)
    print(json.dumps({"nodes": {f"enode_{i}": node for i, node in enumerate(nodes)}}))


class SerializeEGraph(ModulePass):
    name = "eqsat-serialize-egraph"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for f in op.walk():
            if isinstance(f, func.FuncOp):
                serialize_egraph_to_json(f)
