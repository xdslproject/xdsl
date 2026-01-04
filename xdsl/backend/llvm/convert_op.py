from llvmlite import ir

from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import cf, llvm
from xdsl.dialects import vector as xvector
from xdsl.ir import Block, Operation, SSAValue


def convert_op(
    op: Operation,
    builder: ir.IRBuilder,
    val_map: dict[SSAValue, ir.Value],
    block_map: dict[Block, ir.Block] | None = None,
):
    match op:
        # Arithmetic Ops
        case llvm.AddOp():
            val_map[op.results[0]] = builder.add(val_map[op.lhs], val_map[op.rhs])
        case llvm.FAddOp():
            val_map[op.results[0]] = builder.fadd(val_map[op.lhs], val_map[op.rhs])
        case llvm.SubOp():
            val_map[op.results[0]] = builder.sub(val_map[op.lhs], val_map[op.rhs])
        case llvm.FSubOp():
            val_map[op.results[0]] = builder.fsub(val_map[op.lhs], val_map[op.rhs])
        case llvm.MulOp():
            val_map[op.results[0]] = builder.mul(val_map[op.lhs], val_map[op.rhs])
        case llvm.FMulOp():
            val_map[op.results[0]] = builder.fmul(val_map[op.lhs], val_map[op.rhs])
        case llvm.UDivOp():
            val_map[op.results[0]] = builder.udiv(val_map[op.lhs], val_map[op.rhs])
        case llvm.SDivOp():
            val_map[op.results[0]] = builder.sdiv(val_map[op.lhs], val_map[op.rhs])
        case llvm.FDivOp():
            val_map[op.results[0]] = builder.fdiv(val_map[op.lhs], val_map[op.rhs])
        case llvm.URemOp():
            val_map[op.results[0]] = builder.urem(val_map[op.lhs], val_map[op.rhs])
        case llvm.SRemOp():
            val_map[op.results[0]] = builder.srem(val_map[op.lhs], val_map[op.rhs])
        case llvm.FRemOp():
            val_map[op.results[0]] = builder.frem(val_map[op.lhs], val_map[op.rhs])

        # Bitwise Ops
        case llvm.ShlOp():
            val_map[op.results[0]] = builder.shl(val_map[op.lhs], val_map[op.rhs])
        case llvm.LShrOp():
            val_map[op.results[0]] = builder.lshr(val_map[op.lhs], val_map[op.rhs])
        case llvm.AShrOp():
            val_map[op.results[0]] = builder.ashr(val_map[op.lhs], val_map[op.rhs])
        case llvm.AndOp():
            val_map[op.results[0]] = builder.and_(val_map[op.lhs], val_map[op.rhs])
        case llvm.OrOp():
            val_map[op.results[0]] = builder.or_(val_map[op.lhs], val_map[op.rhs])
        case llvm.XOrOp():
            val_map[op.results[0]] = builder.xor(val_map[op.lhs], val_map[op.rhs])

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

        # Casts
        case llvm.TruncOp():
            val_map[op.results[0]] = builder.trunc(
                val_map[op.arg], convert_type(op.results[0].type)
            )
        case llvm.ZExtOp():
            val_map[op.results[0]] = builder.zext(
                val_map[op.arg], convert_type(op.results[0].type)
            )
        case llvm.SExtOp():
            val_map[op.results[0]] = builder.sext(
                val_map[op.arg], convert_type(op.results[0].type)
            )

        case llvm.PtrToIntOp():
            val_map[op.results[0]] = builder.ptrtoint(
                val_map[op.input], convert_type(op.results[0].type)
            )
        case llvm.IntToPtrOp():
            val_map[op.results[0]] = builder.inttoptr(
                val_map[op.input], convert_type(op.results[0].type)
            )

        # Vector Ops
        case xvector.ExtractElementOp():
            val_map[op.results[0]] = builder.extract_element(
                val_map[op.vector], val_map[op.position]
            )
        case xvector.InsertElementOp():
            val_map[op.results[0]] = builder.insert_element(
                val_map[op.dest], val_map[op.source], val_map[op.position]
            )
        case xvector.ShuffleOp():
            mask_list = [int(x) for x in op.mask.iter_values()]
            mask_ty = ir.VectorType(ir.IntType(32), len(mask_list))
            mask_val = ir.Constant(mask_ty, mask_list)
            val_map[op.results[0]] = builder.shuffle_vector(
                val_map[op.v1], val_map[op.v2], mask_val
            )
        case xvector.BitcastOp():
            val_map[op.results[0]] = builder.bitcast(
                val_map[op.source], convert_type(op.results[0].type)
            )
        case xvector.FMAOp():
            val_map[op.results[0]] = builder.fma(
                val_map[op.lhs], val_map[op.rhs], val_map[op.acc]
            )

        # Control Flow
        case cf.BranchOp():
            if block_map and op.successor in block_map:
                builder.branch(block_map[op.successor])
        case cf.ConditionalBranchOp():
            if block_map:
                builder.cbranch(
                    val_map[op.cond], block_map[op.then_block], block_map[op.else_block]
                )
        case cf.SwitchOp():
            if block_map:
                switch = builder.switch(val_map[op.flag], block_map[op.default_block])
                for val, block in zip(op.case_values.iter_values(), op.case_blocks):
                    switch.add_case(val, block_map[block])
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
        case llvm.UnreachableOp():
            builder.unreachable()

        case _:
            raise NotImplementedError(f"Conversion not implemented for op: {op.name}")
