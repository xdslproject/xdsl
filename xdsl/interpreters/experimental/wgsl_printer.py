from __future__ import annotations

from functools import singledispatchmethod
from typing import IO, cast

from xdsl.dialects import arith, builtin, gpu, memref
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.core import Attribute


class WGSLPrinter:
    name_dict: dict[SSAValue, str] = dict()
    count = 0

    def wgsl_name(self, v: SSAValue):
        if v not in self.name_dict:
            if v.name_hint is not None:
                self.name_dict[v] = f"v{v.name_hint}"
            else:
                self.name_dict[v] = f"v{self.count}"
                self.count += 1
        return self.name_dict[v]

    @singledispatchmethod
    def print(self, op: Operation, out_stream: IO[str]) -> None:
        raise NotImplementedError(
            f"Printing of '{op.name}' to WGSL is not implemented yet."
        )

    @print.register
    def _(self, op: gpu.ModuleOp, out_stream: IO[str]):
        for o in op.body.ops:
            if isinstance(o, gpu.FuncOp):
                self.print(o, out_stream)

    @print.register
    def _(self, op: gpu.FuncOp, out_stream: IO[str]):
        for arg in op.body.block.args:
            auth = "read"
            arg_type = ""
            for use in arg.uses:
                if isinstance(use.operation, memref.Store):
                    auth = "read_write"
            if arg.type == builtin.f32:
                arg_type = "f32"
            elif arg.type == builtin.IndexType():
                arg_type = "u32"
            elif isinstance(arg.type, MemRefType):
                memref_typ = cast(MemRefType[Attribute], arg.type)
                arg_type = f"array<{memref_typ.element_type}>"
            arguments = f"""
    @group(0) @binding({arg.index})
    var<storage,{auth}> {self.wgsl_name(arg)}: {arg_type};
"""
            out_stream.write(arguments)

        out_stream.write(
            """
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>) {
"""
        )
        for operation in op.body.ops:
            self.print(operation, out_stream)
        out_stream.write(
            """
            }
            """
        )

    @print.register
    def _(self, op: gpu.ReturnOp, out_stream: IO[str]):
        pass

    @print.register
    def _(self, op: gpu.BlockIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).strip('"')
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = workgroup_id.{dim};"""
        )

    @print.register
    def _(self, op: gpu.ThreadIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).strip('"')
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = local_invocation_id.{dim};"""
        )

    @print.register
    def _(self, op: gpu.GridDimOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).strip('"')
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = num_workgroups.{dim};"""
        )

    @print.register
    def _(self, op: gpu.BlockDimOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).strip('"')
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = local_invocation_id.{dim};"""
        )

    @print.register
    def _(self, op: gpu.GlobalIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).strip('"')
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = global_invocation_id.{dim};"""
        )

    @print.register
    def _(self, op: memref.Load, out_stream: IO[str]):
        memref_type = cast(MemRefType[Attribute], op.memref.type)
        memref_dimension = memref_type.get_num_dims()
        memref_size = memref_type.get_shape()
        load_ref = self.wgsl_name(op.memref)
        name_hint = self.wgsl_name(op.res)
        indices = [self.wgsl_name(i) for i in op.indices]
        index_value = self.calculate_index(memref_dimension, memref_size, indices)
        out_stream.write(
            f"""
        let {name_hint} = {load_ref}[{index_value}];"""
        )

    @print.register
    def _(self, op: memref.Store, out_stream: IO[str]):
        memref_type = cast(MemRefType[Attribute], op.memref.type)
        memref_dimension = memref_type.get_num_dims()
        memref_size = memref_type.get_shape()
        value = self.wgsl_name(op.value)
        store_ref = self.wgsl_name(op.memref)
        indices = [self.wgsl_name(i) for i in op.indices]
        index_value = self.calculate_index(memref_dimension, memref_size, indices)
        out_stream.write(
            f"""
        {store_ref}[{index_value}] = {value};"""
        )

    def calculate_index(
        self, memref_dimension: int, memref_size: tuple[int], indices: list[str]
    ):
        """
        It is used for linearizing known sizes memref accesses.
        """
        for size in memref_size:
            if size == -1:
                raise NotImplementedError(
                    "The WGSL translation only works with known sizes at the moment."
                )
        index_values: list[str] = []
        for i in range(memref_dimension):
            product_of_dims = 1
            for dim in memref_size[i + 1 :]:
                product_of_dims *= dim
            index_values.append(f"{product_of_dims} * {indices[i]}")
        return " + ".join(index_values)

    @print.register
    def _(self, op: gpu.ModuleEndOp, out_stream: IO[str]):
        # Nothing to print :)
        pass

    @print.register
    def _(self, op: arith.Constant, out_stream: IO[str]):
        value = int(str(op.attributes.get("value")).split()[0])
        cons_type = op.result.type
        if isinstance(op.result.type, builtin.IndexType):
            cons_type = "u32"
        name_hint = self.wgsl_name(op.result)
        if cons_type == "u32":
            if value < 0:
                value = 4294967296 + value
            out_stream.write(
                f"""
        let {name_hint} : {cons_type} = {value}u;"""
            )
        else:
            out_stream.write(
                f"""
        let {name_hint} : {cons_type} = {value};"""
            )

    @print.register
    def _(self, op: arith.Addi, out_stream: IO[str]):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        out_stream.write(
            f"""
        let {op_name_hint} = {lhs} + {rhs};"""
        )

    @print.register
    def _(self, op: arith.Muli, out_stream: IO[str]):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        out_stream.write(
            f"""
        let {op_name_hint} = {lhs} * {rhs};"""
        )

    @print.register
    def _(self, op: arith.Subi, out_stream: IO[str]):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        out_stream.write(
            f"""
        let {op_name_hint} = {lhs} - {rhs};"""
        )

    @print.register
    def _(self, op: arith.Mulf, out_stream: IO[str]):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        out_stream.write(
            f"""
        let {op_name_hint} = {lhs} * {rhs};"""
        )

    @print.register
    def _(self, op: arith.Addf, out_stream: IO[str]):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        out_stream.write(
            f"""
        let {op_name_hint} = {lhs} + {rhs};"""
        )

    @print.register
    def _(self, op: arith.Subf, out_stream: IO[str]):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        out_stream.write(
            f"""
        let {op_name_hint} = {lhs} - {rhs};"""
        )
