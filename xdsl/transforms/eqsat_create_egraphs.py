from ordered_set import OrderedSet

from xdsl.context import Context
from xdsl.dialects import builtin, equivalence, func
from xdsl.ir import Block, OpResult, Region, SSAValue
from xdsl.passes import ModulePass


class EqsatCreateEgraphsPass(ModulePass):
    """
    Create an egraph from a function by inserting an `equivalence.graph` operation.

    Input example:
    ```
    func.func @test(%a : index, %b : index) -> (index) {
        %c = arith.addi %a, %b : index
        func.return %c : index
    }
    ```
    Output example:
    ```
    func.func @test(%a : index, %b : index) -> index {
        %c = equivalence.graph -> index {
            %a_1 = equivalence.class %a : index
            %b_1 = equivalence.class %b : index
            %c_1 = arith.addi %a_1, %b_1 : index
            %c_2 = equivalence.class %c_1 : index
            equivalence.yield %c_2 : index
        }
        func.return %c : index
    }
    ```
    """

    name = "eqsat-create-egraphs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for f in op.body.block.ops:
            if isinstance(f, func.FuncOp):
                insert_egraph_op(f)


def insert_egraph_op(f: func.FuncOp):
    egraph_block = Block()
    egraph_body = Region(egraph_block)
    egraph_values: OrderedSet[OpResult] = OrderedSet(())

    def create_eclass(val: SSAValue):
        eclass_op = equivalence.ClassOp(val)
        egraph_block.add_op(eclass_op)
        new_val = eclass_op.results[0]
        egraph_values.add(new_val)
        val.replace_by_if(new_val, lambda u: u.operation is not eclass_op)
        return new_val

    for arg in f.body.block.args:
        create_eclass(arg)

    # we don't walk recursively over all operations, but only
    # the top-level operations in the function body:
    for op in f.body.block.ops:
        if isinstance(op, func.ReturnOp) or not all(
            operand in egraph_values for operand in op.operands
        ):
            continue
        op.detach()
        egraph_block.add_op(op)
        for res in op.results:
            assert res not in egraph_values
            create_eclass(res)

    # Find values that have uses outside the egraph body
    values_to_yield: list[SSAValue] = []
    for val in egraph_values:
        has_external_use = False
        for use in val.uses:
            # Check if the use is outside the egraph block
            if use.operation.parent != egraph_block:
                has_external_use = True
                break
        if has_external_use:
            values_to_yield.append(val)

    # Each value in the egraph block that has a use outside the egraph body should be yielded by the egraph op.
    # Next, these outside uses need to be replaced by the results of the egraph op.
    egraph_block.add_op(equivalence.YieldOp(*values_to_yield))

    # Create the egraph operation with the types of yielded values
    yielded_types = [val.type for val in values_to_yield]
    egraph_op = equivalence.GraphOp(result_types=yielded_types, body=egraph_body)

    for i, val in enumerate(values_to_yield):
        val.replace_by_if(
            egraph_op.results[i], lambda u: u.operation.parent != egraph_block
        )

    # Insert the egraph operation at the beginning of the function body
    assert f.body.block.first_op is not None, "Function body block is empty"
    f.body.block.insert_op_before(egraph_op, f.body.block.first_op)
