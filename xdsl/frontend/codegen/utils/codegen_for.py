import ast
from xdsl.dialects import affine, builtin

from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.codegen.inserter import OpInserter
from xdsl.ir import Block, Region


def check_for_loop_valid(node: ast.For):
    """Aborts if this loop cannot be lowered to xDSL or MLIR."""

    # Make sure we do not support Python hackery like:
    # for x in xs:      for x1, x2, x3 in xs:
    #   ...         or    ...
    # else:
    #   ...
    if len(node.orelse) > 0:
        raise CodegenException(f"unexpected else clause in for loop on line {node.lineno}")
    if not isinstance(node.target, ast.Name):
        raise CodegenException(f"expected a single induction target variable, found multiple in for loop on line {node.lineno}")

    # In xDSL/MLIR we can only have range-based loops as there is no concept of iterator.
    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range" and 1 <= len(node.iter.args) <= 3:
        return
    raise CodegenException(f"not a range-based loop on line {node.lineno}")


def is_affine_for_loop(node: ast.For) -> bool:
    """Returns true if this for loop is affine."""
    args = node.iter.args
    for arg in node.iter.args:
        if not isinstance(arg, ast.Constant):
            return False
    return True


def codegen_affine_for_loop(inserter: OpInserter, node: ast.For, visit_callback):
    """Gnereates xDSL for affine for loops."""

    # First, proces range arguments which should simply constants.
    args = node.iter.args
    start = 0
    end = 0
    step = 1
    if len(args) == 1:
        end = int(args[0].value)
    elif len(args) == 2:
        start = int(args[0].value)
        end = int(args[1].value)
    else:
        start = int(args[0].value)
        end = int(args[1].value)
        step = int(args[2].value)      

    # Save previous insertion point.
    prev_insertion_point = inserter.ip

    entry_block = Block()
    entry_block.insert_arg(builtin.IndexType(), 0)
    body_region = Region.from_block_list([entry_block])

    # Create affine.for operation and insert it.
    op = affine.For.from_region([], start, end, body_region, step)
    inserter.set_insertion_point_from_block(prev_insertion_point)
    inserter.insert_op(op)
    inserter.set_insertion_point_from_block(entry_block)

    # Generate xDSL for the loop body.
    for stmt in node.body:
        visit_callback(stmt)
    inserter.insert_op(affine.Yield.get())

    # Reset insertion point back. 
    inserter.set_insertion_point_from_block(prev_insertion_point)


def codegen_scf_for_loop(inserter: OpInserter, node: ast.For):
    raise CodegenException("conversion to scf.for is not implemented")
