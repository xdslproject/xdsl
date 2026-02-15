from collections.abc import Callable

from llvmlite import ir
from llvmlite.ir import instructions
from llvmlite.ir.types import Type
from llvmlite.ir.values import Block, Value

from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import llvm
from xdsl.ir import Operation, SSAValue


class CastInstrWithFlags(instructions.CastInstr):
    """
    Custom CastInstr that supports flags (workaround for llvmlite limitation).
    llvmlite's CastInstr doesn't accept or output flags, but we need them for
    operations like 'trunc nsw' and 'zext nneg'.
    """

    def __init__(
        self,
        parent: Block,
        op: str,
        val: Value,
        typ: Type,
        name: str = "",
        flags: tuple[str, ...] | list[str] = (),
    ):
        # Call parent's parent (Instruction) to bypass CastInstr's __init__
        instructions.Instruction.__init__(
            self, parent, typ, op, [val], name=name, flags=flags
        )

    def descr(self, buf: list[str]) -> None:
        """Override descr to include flags in output (like base Instruction)."""
        opname = self.opname
        if self.flags:
            opname = " ".join([opname] + list(self.flags))
        operand = self.operands[0]
        metadata = self._stringify_metadata(leading_comma=True)
        buf.append(
            f"{opname} {operand.type} {operand.get_reference()} to {self.type}{metadata}\n"
        )


def _create_cast_with_flags(
    builder: ir.IRBuilder, opname: str, val: ir.Value, typ: ir.Type, flags: list[str]
) -> ir.Value:
    """
    Create a cast instruction with flags.

    This is a workaround for llvmlite's limitation where the builder's
    cast methods don't support flags.
    """
    instr = CastInstrWithFlags(builder.block, opname, val, typ, name="", flags=flags)
    builder._insert(instr)
    return instr


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
    llvm.SIToFPOp: lambda b: b.sitofp,
}

_CAST_OP_NAMES: dict[type[Operation], str] = {
    llvm.TruncOp: "trunc",
    llvm.ZExtOp: "zext",
    llvm.SExtOp: "sext",
    llvm.PtrToIntOp: "ptrtoint",
    llvm.IntToPtrOp: "inttoptr",
    llvm.BitcastOp: "bitcast",
    llvm.FPExtOp: "fpext",
    llvm.SIToFPOp: "sitofp",
}


def _convert_cast(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    """
    Convert a cast operation, handling flags/properties similar to _convert_binop.
    """
    source_val = val_map[op.operands[0]]
    result_type = convert_type(op.results[0].type)
    flags = []

    # Extract flags based on operation type (mirror _convert_binop pattern)
    match op:
        case llvm.IntegerConversionOpOverflow():  # TruncOp
            if op.overflowFlags:
                # Note: IntegerConversionOpOverflow has overflowFlags as OverflowAttr directly,
                # unlike ArithmeticBinOpOverflow which has IntegerAttr[I32]
                flags = [f.value for f in op.overflowFlags.data]
        case llvm.IntegerConversionOpNNeg():  # ZExtOp
            if op.non_neg:
                flags = ["nneg"]
        case _:
            pass

    # Create instruction with or without flags
    if flags:
        # Use custom instruction creator for operations with flags
        opname = _CAST_OP_NAMES[type(op)]
        val_map[op.results[0]] = _create_cast_with_flags(
            builder, opname, source_val, result_type, flags
        )
    else:
        # Use standard builder method for operations without flags
        op_builder = _CAST_OP_MAP[type(op)]
        val_map[op.results[0]] = op_builder(builder)(source_val, result_type)


_ICMP_PRED_MAP: dict[str, tuple[str, bool]] = {
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


def _convert_icmp(
    op: llvm.ICmpOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    predicate = op.predicate.value.data
    flag = llvm.ICmpPredicateFlag.from_int(predicate)
    pred_str = flag.value
    llvm_pred, is_signed = _ICMP_PRED_MAP[pred_str]

    target_func = builder.icmp_signed if is_signed else builder.icmp_unsigned
    val_map[op.results[0]] = target_func(llvm_pred, val_map[op.lhs], val_map[op.rhs])


def _convert_call(
    op: llvm.CallOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    args = [val_map[arg] for arg in op.args]
    if op.callee is None:
        raise NotImplementedError("Indirect calls not yet implemented")
    callee = builder.module.get_global(op.callee.string_value())
    instruction = builder.call(
        callee,
        args,
        cconv=op.CConv.cconv_name,
        tail=op.TailCallKind.data != "none",
        fastmath=[f.value for f in op.fastmathFlags.data],
    )
    if op.returned:
        val_map[op.returned] = instruction


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
        case llvm.ICmpOp():
            _convert_icmp(op, builder, val_map)
        case llvm.CallOp():
            _convert_call(op, builder, val_map)
        case llvm.InlineAsmOp():
            _convert_inline_asm(op, builder, val_map)
        case llvm.ReturnOp():
            _convert_return(op, builder, val_map)
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
