from collections.abc import Callable
from typing import cast

from llvmlite import ir
from llvmlite.ir import instructions
from llvmlite.ir.instructions import PhiInstr
from llvmlite.ir.types import Type
from llvmlite.ir.values import Block as LLVMBlock
from llvmlite.ir.values import Value

from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import llvm, vector
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
)
from xdsl.dialects.vector import FMAOp
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.utils.type import get_element_type_or_self

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


class CastInstrWithFlags(instructions.CastInstr):
    # workaround: llvmlite's CastInstr doesn't support flags like 'trunc nsw' or 'zext nneg'
    def __init__(
        self,
        parent: LLVMBlock,
        op: str,
        val: Value,
        typ: Type,
        name: str = "",
        flags: tuple[str, ...] | list[str] = (),
    ):
        instructions.Instruction.__init__(
            self, parent, typ, op, [val], name=name, flags=flags
        )

    def descr(self, buf: list[str]) -> None:
        opname = (
            " ".join([self.opname] + list(self.flags)) if self.flags else self.opname
        )
        op = self.operands[0]
        buf.append(
            f"{opname} {op.type} {op.get_reference()} to {self.type}{self._stringify_metadata(leading_comma=True)}\n"
        )


def _convert_cast(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    match op:
        case llvm.IntegerConversionOpOverflow() if op.overflowFlags:
            flags = [f.value for f in op.overflowFlags.data]
        case llvm.IntegerConversionOpNNeg() if op.non_neg:
            flags = ["nneg"]
        case _:
            flags = []

    instr = CastInstrWithFlags(
        builder.block,
        _CAST_OP_NAMES[type(op)],
        val_map[op.operands[0]],
        convert_type(op.results[0].type),
        flags=flags,
    )
    builder._insert(instr)
    val_map[op.results[0]] = instr


_FCMP_CMP_MAP = {"eq": "==", "gt": ">", "ge": ">=", "lt": "<", "le": "<=", "ne": "!="}


def _convert_fcmp(
    op: llvm.FCmpOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    pred_int: int = op.predicate.value.data
    flag = llvm.FCmpPredicateFlag.from_int(pred_int)
    pred = flag.value
    is_ordered = pred[0] == "o"
    key = pred[1:]
    cmpop = _FCMP_CMP_MAP.get(key, pred)
    fn = builder.fcmp_ordered if is_ordered else builder.fcmp_unordered
    val_map[op.results[0]] = fn(cmpop, val_map[op.lhs], val_map[op.rhs])


_UNARY_INTRINSIC_MAP: dict[type[Operation], str] = {
    llvm.FAbsOp: "llvm.fabs",
    llvm.FSqrtOp: "llvm.sqrt",
    llvm.FLogOp: "llvm.log",
}


_BINARY_INTRINSIC_MAP: dict[type[Operation], str] = {
    llvm.VectorFMaxOp: "llvm.maxnum",
}


def _convert_unary_intrinsic(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    operand = val_map[op.operands[0]]
    fn_type = ir.FunctionType(operand.type, [operand.type])
    intrinsic_name = _UNARY_INTRINSIC_MAP[type(op)]
    intrinsic = builder.module.declare_intrinsic(intrinsic_name, fnty=fn_type)
    val_map[op.results[0]] = builder.call(intrinsic, [operand])


def _convert_binary_intrinsic(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    lhs = val_map[op.operands[0]]
    rhs = val_map[op.operands[1]]
    fn_type = ir.FunctionType(lhs.type, [lhs.type, rhs.type])
    intrinsic_name = _BINARY_INTRINSIC_MAP[type(op)]
    intrinsic = builder.module.declare_intrinsic(intrinsic_name, fnty=fn_type)
    val_map[op.results[0]] = builder.call(intrinsic, [lhs, rhs])


def _convert_fneg(
    op: llvm.FNegOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    operand = val_map[op.arg]
    val_map[op.res] = builder.fneg(operand)


def _convert_call(
    op: llvm.CallOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    args = [val_map[arg] for arg in op.args]
    if op.callee is None:
        raise NotImplementedError("Indirect calls not yet implemented")
    callee = builder.module.get_global(op.callee.string_value())
    fastmath = (
        [f.value for f in op.fastmathFlags.data]
        if op.returned is not None
        and isinstance(get_element_type_or_self(op.returned.type), AnyFloat)
        else []
    )
    instruction = builder.call(
        callee,
        args,
        cconv=op.CConv.cconv_name,
        tail=op.TailCallKind.data != "none",
        fastmath=fastmath,
    )
    if op.returned:
        val_map[op.returned] = instruction


def _convert_alloca(
    op: llvm.AllocaOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    alloca_instr = builder.alloca(convert_type(op.elem_type), size=val_map[op.size])
    if op.alignment:
        alloca_instr.align = op.alignment.value.data
    val_map[op.results[0]] = alloca_instr


def _convert_load(
    op: llvm.LoadOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    load_instr = builder.load(val_map[op.ptr], typ=convert_type(op.results[0].type))
    if op.alignment:
        load_instr.align = op.alignment.value.data
    val_map[op.results[0]] = load_instr


def _convert_store(
    op: llvm.StoreOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    store_instr = builder.store(val_map[op.value], val_map[op.ptr])
    if op.alignment:
        store_instr.align = op.alignment.value.data


def _convert_getelementptr(
    op: llvm.GEPOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
) -> None:
    # GEPOp mixes static and dynamic indices (placeholder for SSA values)
    typed_ptr_ty = convert_type(op.elem_type).as_pointer()
    casted_ptr = builder.bitcast(val_map[op.ptr], typed_ptr_ty)

    ssa_indices = iter(op.ssa_indices)
    indices = [
        val_map[next(ssa_indices)]
        if idx == llvm.GEP_USE_SSA_VAL
        else ir.Constant(ir.IntType(32), idx)
        for idx in op.rawConstantIndices.iter_values()
    ]

    val_map[op.results[0]] = builder.gep(
        casted_ptr,
        indices,
        inbounds=op.inbounds is not None,
        source_etype=convert_type(op.elem_type),
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


def _convert_select(
    op: llvm.SelectOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    val_map[op.res] = builder.select(val_map[op.cond], val_map[op.lhs], val_map[op.rhs])


def _convert_br(
    op: llvm.BrOp,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
    block_map: dict[Block, LLVMBlock],
):
    dest = op.successor
    parent = op.parent_block()
    assert parent is not None
    current_block = block_map[parent]
    for arg, val in zip(dest.args, op.arguments):
        phi = val_map[arg]
        assert isinstance(phi, PhiInstr)
        phi.add_incoming(val_map[val], current_block)
    builder.branch(block_map[dest])


def _convert_condbr(
    op: llvm.CondBrOp,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
    block_map: dict[Block, LLVMBlock],
):
    then_block = op.then_block
    else_block = op.else_block
    parent = op.parent_block()
    assert parent is not None
    current_block = block_map[parent]
    for arg, val in zip(then_block.args, op.then_arguments):
        phi = val_map[arg]
        assert isinstance(phi, PhiInstr)
        phi.add_incoming(val_map[val], current_block)
    for arg, val in zip(else_block.args, op.else_arguments):
        phi = val_map[arg]
        assert isinstance(phi, PhiInstr)
        phi.add_incoming(val_map[val], current_block)
    builder.cbranch(val_map[op.cond], block_map[then_block], block_map[else_block])


def _convert_masked_store(
    op: llvm.MaskedStoreOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    value = val_map[op.value]
    ptr = val_map[op.data]
    mask = val_map[op.mask]
    alignment = ir.Constant(ir.IntType(32), op.alignment.value.data)
    fn_type = ir.FunctionType(
        ir.VoidType(), [value.type, ptr.type, alignment.type, mask.type]
    )
    intrinsic = builder.module.declare_intrinsic("llvm.masked.store", fnty=fn_type)
    builder.call(intrinsic, [value, ptr, alignment, mask])


def _convert_fma(
    op: FMAOp,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
):
    lhs = val_map[op.lhs]
    rhs = val_map[op.rhs]
    acc = val_map[op.acc]
    res_type = convert_type(op.res.type)
    assert isinstance(res_type, ir.VectorType)
    assert isinstance(res_type.element, (ir.HalfType, ir.FloatType, ir.DoubleType))
    # declare_intrinsic doesn't support VectorType, build name manually
    name = f"llvm.fma.v{res_type.count}{res_type.element.intrinsic_name}"
    fn_type = ir.FunctionType(res_type, [res_type, res_type, res_type])
    try:
        intrinsic = builder.module.get_global(name)
    except KeyError:
        intrinsic = ir.Function(builder.module, fn_type, name=name)
    val_map[op.res] = builder.call(intrinsic, [lhs, rhs, acc])


def _convert_return(
    op: Operation, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    if op.operands:
        builder.ret(val_map[op.operands[0]])
    else:
        builder.ret_void()


def _convert_addressof(
    op: llvm.AddressOfOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    val_map[op.result] = builder.module.get_global(op.global_name.root_reference.data)


def _convert_broadcast(
    op: vector.BroadcastOp,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
):
    source_val = val_map[op.source]
    vec_type = convert_type(op.vector.type)
    n_lanes = op.vector.type.get_shape()[0]
    undef = ir.Constant(vec_type, ir.Undefined)
    inserted = builder.insert_element(undef, source_val, ir.Constant(ir.IntType(32), 0))
    mask = ir.Constant(ir.VectorType(ir.IntType(32), n_lanes), [0] * n_lanes)
    val_map[op.vector] = builder.shuffle_vector(inserted, undef, mask)


_CONSTANT_VALUE_MAP: dict[type[Attribute], Callable[[Attribute], object]] = {
    DenseIntOrFPElementsAttr: lambda v: list(
        cast(DenseIntOrFPElementsAttr, v).iter_values()
    ),
    IntegerAttr: lambda v: cast(IntegerAttr, v).value.data,
    FloatAttr: lambda v: cast(FloatAttr, v).value.data,
}


def _convert_constant(
    op: llvm.ConstantOp, builder: ir.IRBuilder, val_map: dict[SSAValue, ir.Value]
):
    value = op.value
    try:
        handler = _CONSTANT_VALUE_MAP[type(value)]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported constant attribute type: {type(value)}"
        ) from None
    val_map[op.result] = ir.Constant(convert_type(op.result.type), handler(value))


def convert_op(
    op: Operation,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
    block_map: dict[Block, LLVMBlock] | None = None,
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
        case llvm.ICmpOp():
            _convert_icmp(op, builder, val_map)
        case llvm.FCmpOp():
            _convert_fcmp(op, builder, val_map)
        case op if type(op) in _CAST_OP_NAMES:
            _convert_cast(op, builder, val_map)
        case op if type(op) in _UNARY_INTRINSIC_MAP:
            _convert_unary_intrinsic(op, builder, val_map)
        case op if type(op) in _BINARY_INTRINSIC_MAP:
            _convert_binary_intrinsic(op, builder, val_map)
        case llvm.FNegOp():
            _convert_fneg(op, builder, val_map)
        case llvm.CallOp():
            _convert_call(op, builder, val_map)
        case llvm.AllocaOp():
            _convert_alloca(op, builder, val_map)
        case llvm.LoadOp():
            _convert_load(op, builder, val_map)
        case llvm.StoreOp():
            _convert_store(op, builder, val_map)
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
        case llvm.GEPOp():
            _convert_getelementptr(op, builder, val_map)
        case llvm.InlineAsmOp():
            _convert_inline_asm(op, builder, val_map)
        case llvm.BrOp() if block_map is not None:
            _convert_br(op, builder, val_map, block_map)
        case llvm.CondBrOp() if block_map is not None:
            _convert_condbr(op, builder, val_map, block_map)
        case llvm.UnreachableOp():
            builder.unreachable()
        case llvm.SelectOp():
            _convert_select(op, builder, val_map)
        case llvm.MaskedStoreOp():
            _convert_masked_store(op, builder, val_map)
        case llvm.ReturnOp():
            _convert_return(op, builder, val_map)
        case llvm.ZeroOp():
            val_map[op.res] = ir.Constant(convert_type(op.res.type), None)
        case llvm.AddressOfOp():
            _convert_addressof(op, builder, val_map)
        case FMAOp():
            _convert_fma(op, builder, val_map)
        case vector.BroadcastOp():
            _convert_broadcast(op, builder, val_map)
        case llvm.ConstantOp():
            _convert_constant(op, builder, val_map)
        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
