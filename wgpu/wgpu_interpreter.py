from __future__ import annotations

from functools import singledispatchmethod
from typing import cast, IO

from xdsl.dialects import builtin
from xdsl.dialects import gpu, func, memref, arith, cf
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.core import Attribute


class WGPUFunctions:
    name_dict: dict[SSAValue, str] = dict()
    count = 0

    def wgsl_name(self, v: SSAValue):
        if not v in self.name_dict.keys():
            if v.name_hint is not None:
                self.name_dict[v] = f"v{v.name_hint}"
            else:
                self.name_dict[v] = f"v{self.count}"
                self.count += 1
        return self.name_dict[v]

    @singledispatchmethod
    def print(self, op: Operation, out_stream: IO[str]) -> None:
        raise NotImplementedError(
            f"Printing the {op.name} to WGSL is not implemented yet."
        )

    @print.register
    def _(self, op: gpu.ModuleOp, out_stream: IO[str]):
        # print(f"Thats a module : {op}")
        for o in op.body.ops:
            if isinstance(o, func.FuncOp):
                self.print(o, out_stream)
            elif isinstance(o, gpu.FuncOp):
                self.print(o, out_stream)

    @print.register
    def _(self, op: gpu.FuncOp, out_stream: IO[str]):
        print(f"Thats a gpu func : {op}")
        for arg in op.body.block.args:
            auth = "read"
            arg_type = ""
            for use in arg.uses:
                if isinstance(use.operation, memref.Store):
                    auth = "read_write"
            if isinstance(arg.type, builtin.Float32Type):
                arg_type = "f32"
            elif isinstance(arg.type, builtin.IndexType):
                arg_type = "u32"
            elif isinstance(arg.type, MemRefType):
                memref_typ = cast(MemRefType[Attribute], arg.type)
                arg_type = f"array<{memref_typ.element_type}>"
            arguments = f"""
    @group(0) @binding({arg.index})
    var<storage,{auth}> v{arg.name_hint}: {arg_type};
            """
            out_stream.write(arguments)

        out_stream.write(
            f"""
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>, 
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>) {{
        """
        )
        for operation in op.body.ops:
            self.print(operation, out_stream)
        out_stream.write(
            f"""
            }}
            """
        )

    @print.register
    def _(self, op: gpu.ReturnOp, out_stream: IO[str]):
        pass

    @print.register
    def _(self, op: gpu.BlockIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = workgroup_id.{dim};"""
        )

    @print.register
    def _(self, op: gpu.ThreadIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = local_invocation_id.{dim};"""
        )

    @print.register
    def _(self, op: gpu.GridDimOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = num_workgroups.{dim};"""
        )

    @print.register
    def _(self, op: gpu.BlockDimOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = local_invocation_id.{dim};"""
        )

    @print.register
    def _(self, op: gpu.GlobalIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = self.wgsl_name(op.result)
        out_stream.write(
            f"""
        let {name_hint}: u32 = global_invocation_id.{dim};"""
        )

    @print.register
    def _(self, op: memref.Load, out_stream: IO[str]):
        load_ref = op.memref.name_hint
        name_hint = self.wgsl_name(op.res)
        index = ", ".join(self.wgsl_name(i) for i in op.indices)
        out_stream.write(
            f"""
        let {name_hint} = v{load_ref}[{index}];"""
        )

    @print.register
    def _(self, op: memref.Store, out_stream: IO[str]):
        value = self.wgsl_name(op.value)
        store_ref = op.memref.name_hint
        index = ", ".join(self.wgsl_name(i) for i in op.indices)
        out_stream.write(
            f"""
        v{store_ref}[{index}] = {value};"""
        )

    @print.register
    def _(self, op: gpu.ModuleEndOp, out_stream: IO[str]):
        # Nothing to print :)
        pass

    @print.register
    def _(self, op: builtin.ModuleOp, out_stream: IO[str]):
        # Just print the content
        for o in op.ops:
            self.print(o, out_stream)

    @print.register
    def _(self, op: arith.Constant, out_stream: IO[str]):
        value = int(str(op.attributes.get("value")).split()[0])
        cons_type = op.result.type
        if op.result.type.name == "index":
            cons_type = "u32"
        name_hint = self.wgsl_name(op.result)
        if cons_type == "u32":
            if value == -1:
                value = 4294967295
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
    def _(self, op: cf.Branch, out_stream: IO[str]):
        pass
