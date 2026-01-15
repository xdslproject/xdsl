from llvmlite import ir

from xdsl.dialects import llvm
from xdsl.ir import Operation, SSAValue


def _convert_return(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    if op.operands:
        builder.ret(val_map[op.operands[0]])
    else:
        builder.ret_void()


def convert_op(
    op: Operation,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
):
    """
    Convert an xDSL operation to an llvmlite LLVM IR.

    Side effects:
        Mutates val_map by adding entries for the operation's results.

    Args:
        op: The xDSL operation to convert
        builder: The LLVM IR builder for constructing instructions
        val_map: The Mapping from xDSL SSA values to LLVM IR values.
                 This dictionary is mutated to store the LLVM IR value produced by this operation for
                 use by subsequent operations.

    Raises:
        NotImplementedError: If the operation is not supported.
    """
    match op:
        case llvm.ReturnOp() | _ if op.name == "llvm.return":
            _convert_return(op, builder, val_map)
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
