from collections.abc import Callable

from llvmlite import ir

from xdsl.dialects import llvm
from xdsl.ir import Operation, SSAValue

_BINARY_OP_MAP: dict[
    type[Operation], Callable[[ir.IRBuilder], Callable[[ir.Value, ir.Value], ir.Value]]
] = {
    llvm.AddOp: lambda b: b.add,
    llvm.FAddOp: lambda b: b.fadd,
    llvm.SubOp: lambda b: b.sub,
    llvm.FSubOp: lambda b: b.fsub,
    llvm.MulOp: lambda b: b.mul,
    llvm.FMulOp: lambda b: b.fmul,
    llvm.UDivOp: lambda b: b.udiv,
    llvm.SDivOp: lambda b: b.sdiv,
    llvm.FDivOp: lambda b: b.fdiv,
    llvm.URemOp: lambda b: b.urem,
    llvm.SRemOp: lambda b: b.srem,
    llvm.FRemOp: lambda b: b.frem,
    llvm.ShlOp: lambda b: b.shl,
    llvm.LShrOp: lambda b: b.lshr,
    llvm.AShrOp: lambda b: b.ashr,
    llvm.AndOp: lambda b: b.and_,
    llvm.OrOp: lambda b: b.or_,
    llvm.XOrOp: lambda b: b.xor,
}


def _convert_icmp(
    op: llvm.ICmpOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    icmp_pred_map: dict[str, tuple[str, bool]] = {
        "eq": ("==", True),
        "ne": ("!=", True),
        "slt": ("<", True),
        "sle": ("<=", True),
        "ult": ("<", False),
        "ule": ("<=", False),
        "sgt": (">", True),
        "sge": (">=", True),
        "ugt": (">", False),
        "uge": (">=", False),
    }

    predicate = op.predicate.value.data
    flag = llvm.ICmpPredicateFlag.from_int(predicate)
    pred_str = flag.value
    llvm_pred, is_signed = icmp_pred_map[pred_str]

    target_func = builder.icmp_signed if is_signed else builder.icmp_unsigned
    val_map[op.results[0]] = target_func(llvm_pred, val_map[op.lhs], val_map[op.rhs])


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
    if (op_builder := _BINARY_OP_MAP.get(type(op))) is not None:
        val_map[op.results[0]] = op_builder(builder)(
            val_map[op.operands[0]], val_map[op.operands[1]]
        )
        return

    match op:
        case llvm.ICmpOp():
            _convert_icmp(op, builder, val_map)
        case llvm.ReturnOp():
            _convert_return(op, builder, val_map)
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
