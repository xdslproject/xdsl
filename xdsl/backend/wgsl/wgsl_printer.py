from __future__ import annotations

from functools import singledispatchmethod
from typing import cast

from xdsl.dialects import arith, builtin, gpu, memref
from xdsl.dialects.builtin import MemRefType
from xdsl.ir import Operation, SSAValue
from xdsl.utils.base_printer import BasePrinter
from xdsl.utils.hints import isa


class WGSLPrinter(BasePrinter):
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
    def print(self, op: Operation) -> None:
        raise NotImplementedError(
            f"Printing of '{op.name}' to WGSL is not implemented yet."
        )

    @print.register
    def _(self, op: gpu.ModuleOp):
        for o in op.body.ops:
            if isinstance(o, gpu.FuncOp):
                self.print(o)

    @print.register
    def _(self, op: gpu.FuncOp):
        workgroup_size = (1,)
        if op.known_block_size:
            workgroup_size = op.known_block_size.get_values()
        for arg in op.body.block.args:
            auth = "read"
            arg_type = ""
            for use in arg.uses:
                if isinstance(use.operation, memref.StoreOp):
                    auth = "read_write"
            if arg.type == builtin.f32:
                arg_type = "f32"
            elif arg.type == builtin.IndexType():
                arg_type = "u32"
            elif isa(arg.type, MemRefType):
                if arg.type.element_type == builtin.IndexType():
                    arg_type = "u32"
                else:
                    arg_type = arg.type.element_type
                arg_type = f"array<{arg_type}>"
            arguments = f"""
    @group(0) @binding({arg.index})
    var<storage,{auth}> {self.wgsl_name(arg)}: {arg_type};
"""
            self.print_string(arguments)

        self.print_string(
            f"""
    @compute
    @workgroup_size({",".join(str(i) for i in workgroup_size)})
    fn {op.sym_name.data}(@builtin(global_invocation_id) global_invocation_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>) {{
"""
        )
        for operation in op.body.ops:
            self.print(operation)
        self.print_string(
            """
            }
            """
        )

    @print.register
    def _(self, op: gpu.ReturnOp):
        pass

    @print.register
    def _(self, op: gpu.BlockIdOp):
        dim = str(op.dimension.data).strip('"')
        name_hint = self.wgsl_name(op.result)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {name_hint}: u32 = workgroup_id.{dim};")

    @print.register
    def _(self, op: gpu.ThreadIdOp):
        dim = str(op.dimension.data).strip('"')
        name_hint = self.wgsl_name(op.result)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {name_hint}: u32 = local_invocation_id.{dim};")

    @print.register
    def _(self, op: gpu.GridDimOp):
        dim = str(op.dimension.data).strip('"')
        name_hint = self.wgsl_name(op.result)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {name_hint}: u32 = num_workgroups.{dim};")

    @print.register
    def _(self, op: gpu.BlockDimOp):
        dim = str(op.dimension.data).strip('"')
        name_hint = self.wgsl_name(op.result)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {name_hint}: u32 = local_invocation_id.{dim};")

    @print.register
    def _(self, op: gpu.GlobalIdOp):
        dim = str(op.dimension.data).strip('"')
        name_hint = self.wgsl_name(op.result)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {name_hint}: u32 = global_invocation_id.{dim};")

    @print.register
    def _(self, op: memref.LoadOp):
        load_ref = self.wgsl_name(op.memref)
        name_hint = self.wgsl_name(op.res)
        indices = [self.wgsl_name(i) for i in op.indices]
        index_value = self.calculate_index(op, indices)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {name_hint} = {load_ref}[{index_value}];")

    @print.register
    def _(self, op: memref.StoreOp):
        value = self.wgsl_name(op.value)
        store_ref = self.wgsl_name(op.memref)
        indices = [self.wgsl_name(i) for i in op.indices]
        index_value = self.calculate_index(op, indices)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"{store_ref}[{index_value}] = {value};")

    def calculate_index(self, op: memref.StoreOp | memref.LoadOp, indices: list[str]):
        """
        It is used for linearizing known sizes memref accesses.
        """
        memref_type = cast(MemRefType, op.memref.type)
        memref_dimension = memref_type.get_num_dims()
        memref_size = memref_type.get_shape()
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
            index_values.append(f"{product_of_dims}u * {indices[i]}")
        return " + ".join(index_values)

    @print.register
    def _(self, op: arith.ConstantOp):
        value = int(str(op.value).split()[0])
        cons_type = op.result.type
        if isinstance(op.result.type, builtin.IndexType):
            cons_type = "u32"
        name_hint = self.wgsl_name(op.result)
        self.print_string("\n")
        with self.indented(2):
            if cons_type == "u32":
                if value < 0:
                    value = 4294967296 + value
                self.print_string(f"let {name_hint} : {cons_type} = {value}u;")
            else:
                self.print_string(f"let {name_hint} : {cons_type} = {value};")

    @print.register
    def _(self, op: arith.AddiOp):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {op_name_hint} = {lhs} + {rhs};")

    @print.register
    def _(self, op: arith.MuliOp):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {op_name_hint} = {lhs} * {rhs};")

    @print.register
    def _(self, op: arith.SubiOp):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {op_name_hint} = {lhs} - {rhs};")

    @print.register
    def _(self, op: arith.MulfOp):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {op_name_hint} = {lhs} * {rhs};")

    @print.register
    def _(self, op: arith.AddfOp):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {op_name_hint} = {lhs} + {rhs};")

    @print.register
    def _(self, op: arith.SubfOp):
        op_name_hint = self.wgsl_name(op.result)
        lhs = self.wgsl_name(op.lhs)
        rhs = self.wgsl_name(op.rhs)
        self.print_string("\n")
        with self.indented(2):
            self.print_string(f"let {op_name_hint} = {lhs} - {rhs};")
