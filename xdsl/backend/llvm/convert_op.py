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

_CAST_OP_MAP: dict[
    type[Operation], Callable[[ir.IRBuilder], Callable[[ir.Value], ir.Value]]
] = {
    llvm.TruncOp: lambda b: b.trunc,
    llvm.ZExtOp: lambda b: b.zext,
    llvm.SExtOp: lambda b: b.sext,
    llvm.PtrToIntOp: lambda b: b.ptrtoint,
    llvm.IntToPtrOp: lambda b: b.inttoptr,
    llvm.BitcastOp: lambda b: b.bitcast,
    llvm.FPExtOp: lambda b: b.fpext,
}


def _convert_getelementptr(
    op: llvm.GEPOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
) -> None:
    # GEPOp mixes static and dynamic indices
    indices: list[ir.Value] = []
    for idx in op.rawConstantIndices.iter_values():
        if idx == llvm.GEP_USE_SSA_VAL:
            indices.append(val_map[next(iter(op.ssa_indices))])
        else:
            indices.append(ir.Constant(ir.IntType(32), idx))

    typed_ptr_ty = convert_type(op.elem_type).as_pointer()
    casted_ptr = builder.bitcast(val_map[op.ptr], typed_ptr_ty)

    val_map[op.results[0]] = builder.gep(
        casted_ptr, indices, inbounds=op.inbounds is not None
    )


def _convert_call(
    op: llvm.CallOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    args = [val_map[arg] for arg in op.args]
    if op.callee is None:
        raise NotImplementedError("Indirect calls not yet implemented")
    callee = builder.module.get_global(op.callee.string_value())
    res = builder.call(callee, args)
    if op.returned:
        val_map[op.returned] = res


def _convert_return(
    op: llvm.ReturnOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    if op.operands:
        builder.ret(val_map[op.operands[0]])
    else:
        builder.ret_void()


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


def _convert_icmp(
    op: llvm.ICmpOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    icmp_pred_map: dict[str, tuple[str, bool]] = {
        "eq": ("==", True),
        "ne": ("!=", True),
        "slt": ("<", True),
        "sle": ("<=", True),
        "sgt": (">", True),
        "sge": (">=", True),
        "ult": ("<", False),
        "ule": ("<=", False),
        "ugt": (">", False),
        "uge": (">=", False),
    }

    predicate = op.predicate.value.data
    flag = llvm.ICmpPredicateFlag.from_int(predicate)
    pred_str = flag.value
    llvm_pred, is_signed = icmp_pred_map[pred_str]

    target_func = builder.icmp_signed if is_signed else builder.icmp_unsigned
    val_map[op.results[0]] = target_func(llvm_pred, val_map[op.lhs], val_map[op.rhs])


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
    if type(op) in _BINARY_OP_MAP:
        target_func = _BINARY_OP_MAP[type(op)](builder)
        val_map[op.results[0]] = target_func(
            val_map[op.operands[0]], val_map[op.operands[1]]
        )
        return

    if type(op) in _CAST_OP_MAP:
        target_func = _CAST_OP_MAP[type(op)](builder)
        val_map[op.results[0]] = target_func(
            val_map[op.operands[0]], convert_type(op.results[0].type)
        )
        return

    match op:
        case llvm.AllocaOp():
            val_map[op.results[0]] = builder.alloca(
                convert_type(op.elem_type), size=val_map[op.size]
            )
        case llvm.LoadOp():
            val_map[op.results[0]] = builder.load(val_map[op.ptr])
        case llvm.StoreOp():
            builder.store(val_map[op.value], val_map[op.ptr])
        case llvm.GEPOp():
            _convert_getelementptr(op, builder, val_map)
        case llvm.ExtractValueOp():
            val_map[op.results[0]] = builder.extract_value(
                val_map[op.container], list(op.position.iter_values())
            )
        case llvm.InsertValueOp():
            val_map[op.results[0]] = builder.insert_value(
                val_map[op.container],
                val_map[op.value],
                list(op.position.iter_values()),
            )
        case llvm.CallOp():
            _convert_call(op, builder, val_map)
        case llvm.ReturnOp() | _ if op.name == "llvm.return":
            _convert_return(op, builder, val_map)
        case llvm.InlineAsmOp():
            _convert_inline_asm(op, builder, val_map)
        case llvm.ICmpOp():
            _convert_icmp(op, builder, val_map)
        case llvm.UnreachableOp():
            builder.unreachable()
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
