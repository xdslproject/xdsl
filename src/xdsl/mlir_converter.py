import mlir.ir as ir
import array
from xdsl.dialects.builtin import *
from xdsl.dialects.memref import MemRefType


class MLIRConverter:

    def __init__(self, ctx, mlir_module=None):
        self.ctx = ctx
        self.op_to_mlir_ops: Dict[Operation, ir.Operation] = dict()
        self.block_to_mlir_blocks: Dict[Block, ir.Block] = dict()
        self.mlir = mlir_module if mlir_module else ir

    def register_external_dialects(self):
        pass

    def convert_function_type(self, typ: FunctionType) -> ir.FunctionType:
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
        return self.mlir.FunctionType.get(inputs, outputs)

    def convert_type(self, typ: Attribute) -> ir.Type:
        if isinstance(typ, Float32Type):
            return self.mlir.F32Type.get()
        if isinstance(typ, IntegerType):
            return self.mlir.IntegerType.get_signless(typ.width.data)
        if isinstance(typ, IndexType):
            return self.mlir.IndexType.get()
        if isinstance(typ, FunctionType):
            return self.convert_function_type(typ)
        if isinstance(typ, MemRefType):
            return self.mlir.MemRefType.get(
                typ.get_shape(), self.convert_type(typ.element_type))
        if isinstance(typ, TupleType):
            return self.mlir.TupleType.get_tuple(
                [self.convert_type(t) for t in typ.types.data])
        raise Exception(f"Unsupported type for mlir conversion: {typ}")

    def convert_value(self, value: SSAValue) -> ir.Value:
        if isinstance(value, OpResult):
            mlir_op = self.op_to_mlir_ops[value.op]
            return mlir_op.results[value.result_index]
        elif isinstance(value, BlockArgument):
            mlir_block = self.block_to_mlir_blocks[value.block]
            return mlir_block.arguments[value.index]
        raise Exception("Unknown value")

    def convert_attribute(self, attr: Attribute) -> ir.Attribute:
        if isinstance(attr, StringAttr):
            return self.mlir.StringAttr.get(attr.data)
        if isinstance(attr, IntegerAttr):
            return self.mlir.IntegerAttr.get(
                self.convert_type(attr.parameters[1]), attr.parameters[0].data)
        if isinstance(attr, ArrayAttr):
            return self.mlir.ArrayAttr.get(
                [self.convert_attribute(sub_attr) for sub_attr in attr.data])
        if isinstance(attr, DenseIntOrFPElementsAttr):
            # TODO fix this as soon as DEneIntElementsAttr allow us to pass
            #   vector or tensor
            assert (attr.type.get_num_dims() == 1)
            element_type = self.convert_type(attr.type.element_type)
            typ = "vector" if isinstance(attr.type, VectorType) else "tensor"

            return self.mlir.DenseIntElementsAttr.parse(
                f"dense<{[d.parameters[0].data for d in attr.data.data]}> : {typ}<{len(attr.data.data)}x{element_type}>"
            )
        if isinstance(attr, FlatSymbolRefAttr):
            return self.mlir.FlatSymbolRefAttr.get(attr.parameters[0].data)
        # SymbolNameAttrs are in fact just StringAttrs
        if isinstance(attr, SymbolNameAttr):
            return self.mlir.StringAttr.get(attr.parameters[0].data)
        try:
            return self.mlir.TypeAttr.get(self.convert_type(attr))
        except Exception:
            raise Exception(
                f"Unsupported attribute for mlir conversion: {attr}")

    def convert_op(self, op: Operation) -> ir.Operation:
        result_types = [self.convert_type(result.typ) for result in op.results]
        operands = [self.convert_value(operand) for operand in op.operands]
        attributes = {
            name: self.convert_attribute(attr)
            for name, attr in op.attributes.items()
        }

        successors = [
            self.block_to_mlir_blocks[succ] for succ in op.successors
        ]

        mlir_op = self.mlir.Operation.create(op.name,
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

    def convert_block(self, block: Block, mlir_block: ir.Block) -> None:
        ip = self.mlir.InsertionPoint.at_block_begin(mlir_block)
        for op in block.ops:
            ip.insert(self.convert_op(op))
        return mlir_block

    def convert_region(self, region: Region, mlir_region: ir.Region) -> None:
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

    def convert_module(self, op: Operation) -> ir.Module:
        with self.mlir.Context() as mlir_ctx:
            mlir_ctx.allow_unregistered_dialects = True
            self.register_external_dialects()
            with self.mlir.Location.unknown(mlir_ctx):
                if not isinstance(op, ModuleOp):
                    raise Exception("top-level operation should be a ModuleOp")
                mlir_module = self.mlir.Module.create()
                mlir_block = mlir_module.operation.regions[0].blocks[0]
                block = op.regions[0].blocks[0]

                ip = self.mlir.InsertionPoint.at_block_begin(mlir_block)
                for op in block.ops:
                    ip.insert(self.convert_op(op))
                return mlir_module
