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


_ICMP_PRED_MAP = {
    "eq": "==",
    "ne": "!=",
    "slt": "<",
    "sle": "<=",
    "sgt": ">",
    "sge": ">=",
    "ult": "<",
    "ule": "<=",
    "ugt": ">",
    "uge": ">=",
}


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
    is_binary_op = type(op) in _BINARY_OP_MAP
    if is_binary_op:
        target_func = _BINARY_OP_MAP[type(op)](builder)
        val_map[op.results[0]] = target_func(
            val_map[op.operands[0]], val_map[op.operands[1]]
        )
        return

    is_cast_op = type(op) in _CAST_OP_MAP
    if is_cast_op:
        target_func = _CAST_OP_MAP[type(op)](builder)
        val_map[op.results[0]] = target_func(
            val_map[op.operands[0]], convert_type(op.results[0].type)
        )
        return

    match op:
        # Memory Ops
        case llvm.AllocaOp():
            ptr = builder.alloca(convert_type(op.elem_type), size=val_map[op.size])
            val_map[op.results[0]] = ptr
        case llvm.LoadOp():
            val_map[op.results[0]] = builder.load(val_map[op.ptr])
        case llvm.StoreOp():
            builder.store(val_map[op.value], val_map[op.ptr])
        case llvm.GEPOp():
            # GEPOp mixes static and dynamic indices
            indices = []
            ssa_indices_iter = iter(op.ssa_indices)
            for idx in op.rawConstantIndices.iter_values():
                if idx == llvm.GEP_USE_SSA_VAL:
                    indices.append(val_map[next(ssa_indices_iter)])
                else:
                    indices.append(ir.Constant(ir.IntType(32), idx))

            typed_ptr_ty = convert_type(op.elem_type).as_pointer()
            casted_ptr = builder.bitcast(val_map[op.ptr], typed_ptr_ty)

            val_map[op.results[0]] = builder.gep(
                casted_ptr, indices, inbounds=op.inbounds is not None
            )
        case llvm.ExtractValueOp():
            val_map[op.results[0]] = builder.extract_value(
                val_map[op.container], [i for i in op.position.iter_values()]
            )
        case llvm.InsertValueOp():
            val_map[op.results[0]] = builder.insert_value(
                val_map[op.container],
                val_map[op.value],
                [i for i in op.position.iter_values()],
            )
        # Control Flow
        case llvm.ReturnOp() | _ if op.name == "llvm.return":
            if op.operands:
                builder.ret(val_map[op.operands[0]])
            else:
                builder.ret_void()
        case llvm.CallOp():
            args = [val_map[arg] for arg in op.args]
            if op.callee:
                callee = builder.module.get_global(op.callee.string_value())
                res = builder.call(callee, args)
            else:
                raise NotImplementedError("Indirect calls not yet implemented")

            if op.returned:
                val_map[op.returned] = res

        case llvm.InlineAsmOp():
            input_types = [convert_type(arg.type) for arg in op.operands_]
            if op.res:
                ret_type = convert_type(op.res.type)
            else:
                ret_type = ir.VoidType()

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

        case llvm.ICmpOp():
            predicate = op.predicate.value.data
            flag = llvm.ICmpPredicateFlag.from_int(predicate)
            pred_str = flag.value
            llvm_pred = _ICMP_PRED_MAP[pred_str]
            if pred_str in ("eq", "ne", "slt", "sle", "sgt", "sge"):
                val_map[op.results[0]] = builder.icmp_signed(
                    llvm_pred, val_map[op.lhs], val_map[op.rhs]
                )
            else:
                val_map[op.results[0]] = builder.icmp_unsigned(
                    llvm_pred, val_map[op.lhs], val_map[op.rhs]
                )

        case llvm.UnreachableOp():
            builder.unreachable()

        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
