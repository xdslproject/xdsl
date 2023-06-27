from __future__ import annotations

from functools import singledispatchmethod
from typing import cast, IO

from xdsl.dialects import builtin
from xdsl.dialects import gpu, func, memref, arith, cf
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation


class WGPUFunctions:
    @singledispatchmethod
    def print(self, op: Operation, out_stream: IO[str]):
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
            # memref_typ = cast(MemRefType, arg.typ)
            # print(arg.typ, memref_typ)
            auth = "read"
            for use in arg.uses:
                match use.operation:
                    case memref.Store():
                        auth = "read_write"
            match arg.typ:
                case builtin.Float32Type():
                    arguments = f"""
                            @group(0) @binding({arg.index})
                            var<storage,{auth}> data{arg.index + 1}: f32;
                        """
                    out_stream.write(arguments)
                case builtin.IndexType():
                    arguments = f"""
                            @group(0) @binding({arg.index})
                            var<storage,{auth}> data{arg.index + 1}: u32;
                        """
                    out_stream.write(arguments)
                case MemRefType():
                    memref_typ = cast(MemRefType, arg.typ)
                    arguments = f"""
                            @group(0) @binding({arg.index})
                            var<storage,{auth}> data{arg.index + 1}: array<{memref_typ.element_type}>;
                        """
                    out_stream.write(arguments)
        out_stream.write(
            f"""
        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
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
    def _(self, op: func.FuncOp, out_stream: IO[str]):
        print(f"Thats a func : {op}")
        for arg in op.body.block.args:
            memref_typ = cast(MemRefType, arg.typ)
            auth = "read"
            for use in arg.uses:
                match use.operation:
                    case memref.Store():
                        auth = "read_write"
                arguments = f"""
    @group(0) @binding({arg.index})
    var<storage,{auth}> data{arg.index + 1}: array<{memref_typ.element_type}>;
                """
            out_stream.write(arguments)
        out_stream.write(
            f"""
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
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
        print(op.name, op.results)
        # out_stream.write(f"""
        # {op.results}""")

    @print.register
    def _(self, op: gpu.BlockIdOp, out_stream: IO[str]):
        print(op.name)
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = op.result.name_hint
        input_type = op.result.typ
        out_stream.write(
            f"""
        let {name_hint}: u32 = {input_type}.{dim};"""
        )

    @print.register
    def _(self, op: gpu.ThreadIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = op.result.name_hint
        input_type = op.result.typ
        out_stream.write(
            f"""
        let {name_hint}: u32 = {input_type}.{dim};"""
        )

    @print.register
    def _(self, op: gpu.GridDimOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = op.result.name_hint
        input_type = op.result.typ
        out_stream.write(
            f"""
        let {name_hint}: u32 = {input_type}.{dim};"""
        )

    @print.register
    def _(self, op: gpu.BlockDimOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = op.result.name_hint
        input_type = op.result.typ
        out_stream.write(
            f"""
        let {name_hint}: u32 = {input_type}.{dim};"""
        )

    @print.register
    def _(self, op: gpu.GlobalIdOp, out_stream: IO[str]):
        dim = str(op.dimension.value.param).replace('"', "")
        name_hint = op.result.name_hint
        input_type = op.result.typ
        out_stream.write(
            f"""
        let {name_hint}: u32 = {input_type}.{dim};"""
        )

    @print.register
    def _(self, op: memref.Load, out_stream: IO[str]):
        load_ref = op.memref.index
        name_hint = op.res.name_hint
        data_load = f"""data{load_ref + 1}"""
        index = op.indices[0].name_hint
        out_stream.write(
            f"""
        let {name_hint} = {data_load}[{index}];"""
        )

    @print.register
    def _(self, op: memref.Store, out_stream: IO[str]):
        value = op.value.name_hint
        store_ref = op.memref.index
        data_store = f"""data{store_ref + 1}"""
        index = op.indices[0].name_hint
        out_stream.write(
            f"""
        {data_store}[{index}] = {value};"""
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
        value = int(str(op.attributes.get('value')).split()[0])
        cons_type = op.result.typ
        if op.result.typ.name == 'index':
            cons_type = 'u32'
        name_hint = op.result.name_hint
        if cons_type == 'u32':
            if value == -1:
                value = 4294967295
            out_stream.write(f"""
        let {name_hint} : {cons_type} = {value}u;""")
        else:
            out_stream.write(f"""
        let {name_hint} : {cons_type} = {value};""")

    @print.register
    def _(self, op: arith.Addi, out_stream: IO[str]):
        print(op.lhs)
        op_name_hint = op.result.name_hint
        lhs = op.lhs.name_hint
        rhs = op.rhs.name_hint
        out_stream.write(f"""
        let {op_name_hint} = {lhs} + {rhs};""")

    @print.register
    def _(self, op: arith.Muli, out_stream: IO[str]):
        op_name_hint = op.result.name_hint
        lhs = op.lhs.name_hint
        rhs = op.rhs.name_hint
        out_stream.write(f"""
        let {op_name_hint} = {lhs} * {rhs};""")

    @print.register
    def _(self, op: arith.Subi, out_stream: IO[str]):
        op_name_hint = op.result.name_hint
        lhs = op.lhs.name_hint
        rhs = op.rhs.name_hint
        out_stream.write(f"""
        let {op_name_hint} = {lhs} - {rhs};""")
    @print.register
    def _(self, op: arith.Mulf, out_stream: IO[str]):
        op_name_hint = op.result.name_hint
        lhs = op.lhs.name_hint
        rhs = op.rhs.name_hint
        out_stream.write(f"""
        let {op_name_hint} = {lhs} * {rhs};""")

    @print.register
    def _(self, op: arith.Addf, out_stream: IO[str]):
        op_name_hint = op.result.name_hint
        lhs = op.lhs.name_hint
        rhs = op.rhs.name_hint
        out_stream.write(f"""
        let {op_name_hint} = {lhs} + {rhs};""")

    @print.register
    def _(self, op: cf.Branch, out_stream: IO[str]):
        pass
