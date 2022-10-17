from xdsl import ensure_mlir_module_loaded

ensure_mlir_module_loaded()

from xdsl import _mlir_module as mlir  # type: ignore

from xdsl.ir import (MLContext, SSAValue, OpResult, Block, Operation, Region,
                     BlockArgument, Attribute)
from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, IntegerAttr,
                                   VectorType, IntegerType, IndexType,
                                   ArrayAttr, FlatSymbolRefAttr, StringAttr,
                                   FunctionType, TupleType, ModuleOp,
                                   Float32Type, SymbolNameAttr, UnitAttr)
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.llvm import LLVMStructType
from typing import Dict


class MLIRConverter:

    def __init__(self, ctx: MLContext):
        self.ctx = ctx
        self.op_to_mlir_ops: Dict[Operation, mlir.ir.Operation] = dict()
        self.block_to_mlir_blocks: Dict[Block, mlir.ir.Block] = dict()

    def register_external_dialects(self):
        pass

    def convert_function_type(self, typ: FunctionType) -> mlir.ir.FunctionType:
        input_array = typ.parameters[0]
        output_array = typ.parameters[1]
        inputs = [
            self.convert_type(input_type_attr)
            for input_type_attr in input_array.data
        ]
        outputs = [
            self.convert_type(output_type_attr)
            for output_type_attr in output_array.data
        ]
        return mlir.ir.FunctionType.get(inputs, outputs)

    def get_mlir_struct_string(self, typ: LLVMStructType) -> str:
        types = []
        for t in typ.types.data:
            types.append(
                self.get_mlir_struct_string(t) if isinstance(
                    t, LLVMStructType) else str(self.convert_type(t)))
        type_string = "(" + ", ".join(types) + ")"

        if typ.struct_name.data == "":
            return "!llvm.struct<" + type_string + ">"
        return "!llvm.struct<" + typ.struct_name.data + ", " + type_string + ">"

    def convert_type(self, typ: Attribute) -> mlir.ir.Type:
        if isinstance(typ, Float32Type):
            return mlir.ir.F32Type.get()
        if isinstance(typ, IntegerType):
            return mlir.ir.IntegerType.get_signless(typ.width.data)
        if isinstance(typ, IndexType):
            return mlir.ir.IndexType.get()
        if isinstance(typ, FunctionType):
            return self.convert_function_type(typ)
        if isinstance(typ, MemRefType):
            return mlir.ir.MemRefType.get(typ.get_shape(),
                                          self.convert_type(typ.element_type))
        if isinstance(typ, TupleType):
            return mlir.ir.TupleType.get_tuple(
                [self.convert_type(t) for t in typ.types.data])
        if isinstance(typ, LLVMStructType):
            mlir_module = mlir.ir.Module.parse(
                "%0 = llvm.mlir.undef : " + self.get_mlir_struct_string(typ))
            return mlir_module.body.operations.__getitem__(
                0).operation.result.type
        raise Exception(f"Unsupported type for mlir conversion: {typ}")

    def convert_value(self, value: SSAValue) -> mlir.ir.Value:
        if isinstance(value, OpResult):
            mlir_op = self.op_to_mlir_ops[value.op]
            return mlir_op.results[value.result_index]
        elif isinstance(value, BlockArgument):
            mlir_block = self.block_to_mlir_blocks[value.block]
            return mlir_block.arguments[value.index]
        raise Exception("Unknown value")

    def convert_attribute(self, attr: Attribute) -> mlir.ir.Attribute:
        if isinstance(attr, StringAttr):
            return mlir.ir.StringAttr.get(attr.data)
        if isinstance(attr, IntegerAttr):
            return mlir.ir.IntegerAttr.get(
                self.convert_type(attr.parameters[1]), attr.parameters[0].data)
        if isinstance(attr, ArrayAttr):
            return mlir.ir.ArrayAttr.get(
                [self.convert_attribute(sub_attr) for sub_attr in attr.data])
        if isinstance(attr, DenseIntOrFPElementsAttr):
            # TODO fix this as soon as DEneIntElementsAttr allow us to pass
            #   vector or tensor
            assert (attr.type.get_num_dims() == 1)
            element_type = self.convert_type(attr.type.element_type)
            typ = "vector" if isinstance(attr.type, VectorType) else "tensor"

            return mlir.ir.DenseIntElementsAttr.parse(
                f"dense<{[d.parameters[0].data for d in attr.data.data]}> : "
                f"{typ}<{len(attr.data.data)}x{element_type}>")
        if isinstance(attr, FlatSymbolRefAttr):
            return mlir.ir.FlatSymbolRefAttr.get(attr.parameters[0].data)
        if isinstance(attr, UnitAttr):
            return mlir.ir.UnitAttr.get()
        # SymbolNameAttrs are in fact just StringAttrs
        if isinstance(attr, SymbolNameAttr):
            return mlir.ir.StringAttr.get(attr.parameters[0].data)
        try:
            return mlir.ir.TypeAttr.get(self.convert_type(attr))
        except Exception:
            raise Exception(
                f"Unsupported attribute for mlir conversion: {attr}")

    def convert_op(self, op: Operation) -> mlir.ir.Operation:
        result_types = [self.convert_type(result.typ) for result in op.results]
        operands = [self.convert_value(operand) for operand in op.operands]
        attributes = {
            name: self.convert_attribute(attr)
            for name, attr in op.attributes.items()
        }

        successors = [
            self.block_to_mlir_blocks[succ] for succ in op.successors
        ]

        mlir_op = mlir.ir.Operation.create(op.name,
                                           results=result_types,
                                           operands=operands,
                                           attributes=attributes,
                                           successors=successors,
                                           regions=len(op.regions))
        self.op_to_mlir_ops[op] = mlir_op

        for region_idx in range(len(op.regions)):
            self.convert_region(op.regions[region_idx],
                                mlir_op.regions[region_idx])
        return mlir_op

    def convert_block(self, block: Block, mlir_block: mlir.ir.Block) -> None:
        ip = mlir.ir.InsertionPoint.at_block_begin(mlir_block)
        for op in block.ops:
            ip.insert(self.convert_op(op))
        return mlir_block

    def convert_region(self, region: Region,
                       mlir_region: mlir.ir.Region) -> None:
        assert (len(mlir_region.blocks) == 0)
        for block in region.blocks:
            mlir_block_args = [
                self.convert_type(block_operand.typ)
                for block_operand in block.args
            ]
            mlir_block = mlir_region.blocks.append(*mlir_block_args)
            self.block_to_mlir_blocks[block] = mlir_block

        for block in region.blocks:
            mlir_block = self.block_to_mlir_blocks[block]
            self.convert_block(block, mlir_block)

    def convert_module_with_ctx(self, op: Operation,
                                mlir_ctx: mlir.ir.Context) -> mlir.ir.Module:
        with mlir.ir.Location.unknown(mlir_ctx):
            if not isinstance(op, ModuleOp):
                raise Exception("top-level operation should be a ModuleOp")
            mlir_module = mlir.ir.Module.create()
            mlir_block = mlir_module.operation.regions[0].blocks[0]
            block = op.regions[0].blocks[0]

            ip = mlir.ir.InsertionPoint.at_block_begin(mlir_block)
            for op in block.ops:
                ip.insert(self.convert_op(op))
            return mlir_module

    def convert_module(self, op: Operation) -> mlir.ir.Module:
        with mlir.ir.Context() as mlir_ctx:
            mlir_ctx.allow_unregistered_dialects = True
            self.register_external_dialects()
            return self.convert_module_with_ctx(op, mlir_ctx)
