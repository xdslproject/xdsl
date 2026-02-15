from collections.abc import Callable

from llvmlite import ir

from xdsl.backend.llvm.convert_type import convert_type
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


def _convert_binop(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    kwargs = {}
    op_builder = _BINARY_OP_MAP[type(op)]

    match op:
        case llvm.ArithmeticBinOpOverflow():
            if op.overflowFlags:
                overflow_attr = llvm.OverflowAttr.from_int(op.overflowFlags.value.data)
                kwargs["flags"] = [f.value for f in overflow_attr.data]
        case llvm.AbstractFloatArithOp():
            if op.fastmathFlags:
                kwargs["flags"] = [f.value for f in op.fastmathFlags.data]
        case llvm.ArithmeticBinOpExact():
            if op.is_exact:
                kwargs["flags"] = ["exact"]
        case llvm.ArithmeticBinOpDisjoint():
            if op.is_disjoint:
                kwargs["flags"] = ["disjoint"]
        case _:
            pass

    val_map[op.results[0]] = op_builder(builder)(
        val_map[op.operands[0]], val_map[op.operands[1]], **kwargs
    )


_CAST_OP_MAP: dict[
    type[Operation], Callable[[ir.IRBuilder], Callable[[ir.Value, ir.Type], ir.Value]]
] = {
    llvm.TruncOp: lambda b: b.trunc,
    llvm.ZExtOp: lambda b: b.zext,
    llvm.SExtOp: lambda b: b.sext,
    llvm.PtrToIntOp: lambda b: b.ptrtoint,
    llvm.IntToPtrOp: lambda b: b.inttoptr,
    llvm.BitcastOp: lambda b: b.bitcast,
    llvm.FPExtOp: lambda b: b.fpext,
}


def _convert_cast(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    op_builder = _CAST_OP_MAP[type(op)]
    val_map[op.results[0]] = op_builder(builder)(
        val_map[op.operands[0]], convert_type(op.results[0].type)
    )


def _convert_inline_asm(
    op: llvm.InlineAsmOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    input_types = [convert_type(arg.type) for arg in op.operands_]
    ret_type = convert_type(op.res.type) if op.res else ir.VoidType()
    ftype = ir.FunctionType(ret_type, input_types)
    asm = ir.InlineAsm(
        ftype,
        op.asm_string.data,
        op.constraints.data,
        side_effect=op.has_side_effects is not None,
    )
    args = [val_map[arg] for arg in op.operands_]
    res = builder.call(asm, args)
    if op.res:
        val_map[op.results[0]] = res


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
        case op if type(op) in _BINARY_OP_MAP:
            _convert_binop(op, builder, val_map)
        case op if type(op) in _CAST_OP_MAP:
            _convert_cast(op, builder, val_map)
        case llvm.InlineAsmOp():
            _convert_inline_asm(op, builder, val_map)
        case llvm.ReturnOp():
            _convert_return(op, builder, val_map)
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
