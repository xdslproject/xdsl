from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, cast
from contextlib import contextmanager

from xdsl.dialects import arith, csl, scf, memref
from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerType,
    DenseIntOrFPElementsAttr,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    Signedness,
    SignednessAttr,
    StringAttr,
    TensorType,
    UnitAttr,
)
from xdsl.ir import Attribute, Block, SSAValue, Region
from xdsl.ir.core import BlockOps, Operation, TypeAttribute
from xdsl.irdl.irdl import Operand


@dataclass
class CslPrintContext:
    _INDENT_SIZE = 2
    _INDEX = "i32"
    DIVIDER = "// >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<< //"
    output: IO[str]

    variables: dict[SSAValue, str] = field(default_factory=dict)

    _counter: int = field(default=0)

    _prefix: str = field(default="")

    _symbols_to_export: dict[str, tuple[TypeAttribute, bool | None]] \
        = field(default_factory=dict)

    def print(self, text: str, prefix: str = "", end: str = "\n"):
        """
        Print `text` line by line, prefixed by self._prefix and prefix.
        """
        for l in text.split("\n"):
            print(self._prefix + prefix + l, file=self.output, end=end)

    @contextmanager
    def _in_block(self, block_name: str):
        self.print(f"{block_name} {{")
        old_prefix = self._prefix
        self._prefix += self._INDENT_SIZE * " "
        yield
        self._prefix = old_prefix
        self.print("}")
        pass

    def _var_use(self, val: SSAValue, intro: str = "const"):
        if val in self.variables:
            return f"{self._get_variable_name_for(val)}"
        else:
            return f"{intro} {self._get_variable_name_for(val)} : {self.mlir_type_to_csl_type(val.type)}"

    def _get_variable_name_for(self, val: SSAValue, hint: str | None = None) -> str:
        """
        Get an assigned variable name for a given SSA Value
        """
        if val in self.variables:
            return self.variables[val]

        taken_names = set(self.variables.values())

        if hint is None:
            hint = val.name_hint

        if hint is not None and hint not in taken_names:
            name = hint
        else:
            prefix = "v" if val.name_hint is None else val.name_hint

            name = f"{prefix}{self._counter}"
            self._counter += 1

            while name in taken_names:
                name = f"{prefix}{self._counter}"
                self._counter += 1

        self.variables[val] = name
        return name

    def mlir_type_to_csl_type(self, type_attr: Attribute) -> str:
        """
        Convert an MLR type to a csl type. CSL supports a very limited set of types:

        - integer types: i16, u16, i32, u32
        - float types: f16, f32
        - pointers: [*]f32
        - arrays: [64]f32

        This method does not yet support all the types and will be expanded as needed later.
        """
        match type_attr:
            case csl.TypeType():
                return "type"
            case csl.PtrType(type=ty, kind=kind, constness=const):
                match kind.data:
                    case csl.PtrKind.SINGLE: sym = "*"
                    case csl.PtrKind.MANY: sym = "[*]"
                match const.data:
                    case csl.PtrConst.CONST: mut = "const "
                    case csl.PtrConst.MUT: mut = ""
                ty = self.mlir_type_to_csl_type(ty)
                return f"{sym}{mut}{ty}"
            case csl.StringType():
                return "comptime_string"
            case csl.ComptimeStructType():
                return "comptime_struct"
            case csl.ColorType():
                return "color"
            case Float16Type():
                return "f16"
            case Float32Type():
                return "f32"
            case IndexType():
                return self._INDEX
            case IntegerType(
                width=IntAttr(data=width),
                signedness=SignednessAttr(data=Signedness.UNSIGNED),
            ):
                return f"u{width}"
            case IntegerType(width=IntAttr(data=width)):
                return "bool" if width == 1 else f"i{width}"
            case FunctionType(inputs=inp, outputs=out) if len(out) <= 1:
                args = map(self.mlir_type_to_csl_type, inp)
                ret = self.mlir_type_to_csl_type(out.data[0]) \
                    if len(out) else "void"
                return f"fn({', '.join(args)}) {ret}"
            case MemRefType() | TensorType():
                t: ContainerType[TypeAttribute] = type_attr
                shape = ", ".join(str(s) for s in t.get_shape())
                type = self.mlir_type_to_csl_type(t.get_element_type())
                return f"[{shape}]{type}"
            case _:
                return f"<!unknown type {type_attr}>"

    def attribute_value_to_str(self, attr: Attribute) -> str:
        """
        Takes a value-carrying attribute (IntegerAttr, FloatAttr, etc.)
        and converts it to a csl expression representing that value literal (0, 3.14, ...)
        """
        match attr:
            case IntAttr(data=val):
                return str(val)
            case IntegerAttr(value=val, type=IntegerType(width=IntAttr(data=width))) if width == 1:
                return str(bool(val.data)).lower()
            case IntegerAttr(value=val):
                return str(val.data)
            case FloatAttr(value=val):
                return str(val.data)
            case StringAttr() as s:
                return f'"{s.data}"'
            case TypeAttribute() as ty:
                return self.mlir_type_to_csl_type(ty)
            case _:
                return f"<!unknown value {attr}>"

    def attribute_type_to_str(self, attr: Attribute) -> str:
        """
        Takes a value-carrying attribute and (IntegerAttr, FloatAttr, etc.)
        and converts it to a csl expression representing the value's type (f32, u16, ...)
        """
        match attr:
            case IntAttr():
                return "<!indeterminate IntAttr type>"
            case csl.ComptimeStructType() | csl.StringType() | csl.TypeType():
                return self.mlir_type_to_csl_type(attr)
            case TypeAttribute():
                return self.mlir_type_to_csl_type(csl.TypeType())
            case StringAttr():
                return self.mlir_type_to_csl_type(csl.StringType())
            case IntegerAttr(type=(IntegerType() | IndexType()) as int_t):
                return self.mlir_type_to_csl_type(int_t)
            case FloatAttr(type=(Float16Type() | Float32Type()) as float_t):
                return self.mlir_type_to_csl_type(float_t)
            case _:
                return f"<!unknown type of {attr}>"

    def _task_or_fn(self, introducer: str, name: StringAttr, bdy: Region, ftyp: FunctionType):
        args = ", ".join(
            f"{self._get_variable_name_for(arg)} : {self.mlir_type_to_csl_type(arg.type)}" for arg in bdy.block.args)
        ret = 'void' if len(ftyp.outputs) == 0 else self.mlir_type_to_csl_type(
            ftyp.outputs.data[0])
        self.print(f"{introducer} {name.data}({args}) {ret} {{")
        self.descend().print_block(bdy.block)
        self.print("}")

    def _wrapp_task_id(self, kind: csl.TaskKind, id: int):
        if kind == csl.TaskKind.DATA:
            return f"@get_color({id})"
        return str(id)

    def _bind_task(self, name: str, kind: csl.TaskKind, id: int):
        with self._in_block("comptime"):
            self.print(
                f"@bind_{kind.value}_task({name}, @get_{kind.value}_task_id({
                    self._wrapp_task_id(kind, id)}));")

    def _ptr_kind_to_introducer(self, kind: csl.PtrConst):
        match kind:
            case csl.PtrConst.CONST: return "const"
            case csl.PtrConst.MUT: return "var"

    def _ptr_kind_from_bool(self, mutable: bool):
        return csl.PtrConst.MUT if mutable else csl.PtrConst.CONST

    def _sym_constness(self, ty: TypeAttribute):
        if isinstance(ty, FunctionType):
            return None
        assert isinstance(ty, csl.PtrType), \
            "Type of the exported symbol has to be PtrType"
        match ty.constness.data:
            case csl.PtrConst.MUT: return True
            case csl.PtrConst.CONST: return False

    def _memref_global_init(self, init: Attribute, type: str):
        match init:
            case UnitAttr():
                return ""
            case DenseIntOrFPElementsAttr():
                data = init.data.data
                assert len(data) == 1, \
                    f"Memref global initialiser has to have 1 value, got {
                        len(data)}"
                return f" = @constants({type}, {self.attribute_value_to_str(data[0])})"
            case other:
                return f"<unknown memref.global init type {other}>"

    def _binop(self, lhs: Operand, rhs: Operand, res: SSAValue, op: str):
        name_lhs = self._get_variable_name_for(lhs)
        name_rhs = self._get_variable_name_for(rhs)
        return f"{self._var_use(res)} = {name_lhs} {op} {name_rhs};"

    def print_block(self, body: Block):
        """
        Walks over a block and prints every operation in the block.
        """
        for op in body.ops:
            self.print_op(op)

    def print_op(self, op: Operation):
        match op:
            case arith.Constant(value=v, result=r) \
                    | csl.ConstStrOp(string=v, res=r)\
                    | csl.ConstTypeOp(type=v, res=r):
                # v is an attribute that "carries a value", e.g. an IntegerAttr or FloatAttr

                # convert the carried value to a csl value
                value_str = self.attribute_value_to_str(v)

                # emit a constant instantiation:
                self.print(f"{self._var_use(r)} = {value_str};")
            case csl.ImportModuleConstOp(module=module, params=params, result=res):
                name = self._get_variable_name_for(res)
                params_str = ""
                if params is not None:
                    params_str = f", {self._get_variable_name_for(params)}"

                self.print(
                    f'const {name} : imported_module = @import_module("{
                        module.data}"{params_str});'
                )
            case csl.MemberCallOp(field=callee, args=args, struct=struct, result=res):
                args = ", ".join(map(self._get_variable_name_for, args))
                struct_str = f"{self._get_variable_name_for(struct)}."
                var = f"{self._var_use(res)} = " if res is not None else ""
                self.print(f"{var}{struct_str}{callee.data}({args});")
            case csl.CallOp(callee=callee, args=args, result=res):
                args = ", ".join(map(self._get_variable_name_for, args))
                var = f"{self._var_use(res)} = " if res is not None else ""
                self.print(f"{var}{callee.string_value()}({args});")
            case csl.MemberAccessOp(struct=struct, field=field, result=res):
                struct_var = self._get_variable_name_for(struct)
                self.print(
                    f"{self._var_use(res)} = {struct_var}.{field.data};"
                )
            case csl.TaskOp(sym_name=name, body=bdy, function_type=ftyp, kind=kind, id=id):
                self._task_or_fn("task", name, bdy, ftyp)
                self._bind_task(name.data, kind.data, id.value.data)
            case csl.FuncOp(sym_name=name, body=bdy, function_type=ftyp):
                self._task_or_fn("fn", name, bdy, ftyp)
            case csl.ReturnOp(ret_val=None):
                self.print("return;")
            case csl.ReturnOp(ret_val=val) if val is not None:
                self.print(f"return {self._get_variable_name_for(val)};")
            case scf.For(lb=lower, ub=upper, step=stp, body=bdy, res=res, iter_args=it):
                idx, *args = bdy.block.args
                for i, r, a in zip(res, it, args):
                    r_name = self._get_variable_name_for(r)
                    self.print(f"{self._var_use(i, "var")} = {r_name};")
                    self.variables[a] = self.variables[i]
                for op in bdy.block.ops:
                    if not isinstance(op, scf.Yield):
                        continue
                    for y, a in zip(op.arguments, args):
                        self.variables[y] = self.variables[a]

                idx_type = self.mlir_type_to_csl_type(idx.type)
                lower_name, upper_name, stp_name, idx_name = map(
                    self._get_variable_name_for, (lower, upper, stp, idx))
                self.print(
                    f"for(@range({idx_type}, {lower_name}, {
                        upper_name}, {stp_name})) |{idx_name}| {{"
                )
                self.descend().print_block(bdy.block)
                self.print("}")
            case csl.LayoutOp(body=bdy):
                with self._in_block("layout"):
                    self.print_block(bdy.block)
                    for name, val in self._symbols_to_export.items():
                        ty = self.attribute_value_to_str(val[0])
                        mut = str(val[1]).lower() \
                            if val[1] is not None else ""
                        self.print(f'@export_name("{name}", {ty}, {mut});')
            case csl.SetTileCodeOp(file=file, x_coord=x_coord, y_coord=y_coord, params=params):
                file = self.attribute_value_to_str(file)
                x = self._get_variable_name_for(x_coord)
                y = self._get_variable_name_for(y_coord)
                params = self._get_variable_name_for(params) \
                    if params else ""
                self.print(
                    f"@set_tile_code({x}, {y}, {file}, {params});")
            case csl.SetRectangleOp(x_dim=x_dim, y_dim=y_dim):
                x = self._get_variable_name_for(x_dim)
                y = self._get_variable_name_for(y_dim)
                self.print(
                    f"@set_rectangle({x}, {y});")
            case csl.SymbolExportOp(value=val, type=ty) as exp:
                name = exp.get_name()
                q_name = f'"{name}"'
                self._symbols_to_export[name] = (
                    ty,
                    self._sym_constness(ty)
                )
                ty = self.attribute_value_to_str(ty)
                if val is not None:
                    val = self._get_variable_name_for(val)
                else:
                    val = name
                with self._in_block("comptime"):
                    self.print(f"@export_symbol({val}, {q_name});")
            case csl.GetColorOp(id=id, res=res):
                id = self.attribute_value_to_str(id)
                self.print(f"{self._var_use(res)} = @get_color({id});")
            case csl.ConstStructOp(items=items, ssa_fields=fields, ssa_values=values, res=res):
                items = items or DictionaryAttr({})
                fields = fields or ArrayAttr([])
                self.print(f"{self._var_use(res)} = .{{")
                for k, v in items.data.items():
                    v = self.attribute_value_to_str(v)
                    self.print(f".{k} = {v},",
                               prefix=self._INDENT_SIZE * " ")
                for k, v in zip(fields.data, values):
                    v = self._get_variable_name_for(v)
                    self.print(f".{k.data} = {v},",
                               prefix=self._INDENT_SIZE * " ")
                self.print("};")
            case csl.ParamOp(init_value=init, param_name=name, res=res):
                init = f" = {
                    self.attribute_value_to_str(init)}" if init else ""
                ty = self.mlir_type_to_csl_type(res.type)
                self.print(f"param {name.data} : {ty}{init};")
            case csl.AddressOfOp(value=val, res=res):
                val_name = self._get_variable_name_for(val)
                use = self._var_use(
                    res,
                    self._ptr_kind_to_introducer(cast(csl.PtrType, res.type).constness.data))
                self.print(f"{use} = &{val_name};")
            case memref.Global(sym_name=name, type=ty, initial_value=init, constant=const):
                name = name.data
                ty = self.mlir_type_to_csl_type(ty)
                init = self._memref_global_init(init, ty)
                var = self._ptr_kind_to_introducer(
                    self._ptr_kind_from_bool(const is None))
                self.print(f"{var} {name} : {ty}{init};")
            case memref.GetGlobal(name_=name, memref=res):
                # We print the array definition when the global is defined
                self.variables[res] = name.string_value()
            case csl.RpcOp(id=id):
                id = self._get_variable_name_for(id)
                with self._in_block("comptime"):
                    self.print(f"@rpc(@get_data_task_id({id}));")
            case arith.IndexCastOp(input=inp, result=res)   \
                    | arith.SIToFPOp(input=inp, result=res) \
                    | arith.FPToSIOp(input=inp, result=res) \
                    | arith.ExtFOp(input=inp, result=res)   \
                    | arith.TruncFOp(input=inp, result=res) \
                    | arith.TruncIOp(input=inp, result=res) \
                    | arith.ExtSIOp(input=inp, result=res)  \
                    | arith.ExtUIOp(input=inp, result=res):
                name_in = self._get_variable_name_for(inp)
                type_out = self.mlir_type_to_csl_type(res.type)
                self.print(
                    f"{self._var_use(res)} = @as({type_out}, {name_in});")
            case arith.Muli(lhs=lhs, rhs=rhs, result=res) \
                    | arith.Mulf(lhs=lhs, rhs=rhs, result=res):
                self.print(self._binop(lhs, rhs, res, "*"))
            case arith.Addi(lhs=lhs, rhs=rhs, result=res) \
                    | arith.Addf(lhs=lhs, rhs=rhs, result=res):
                self.print(self._binop(lhs, rhs, res, "+"))
            case memref.Store(value=val, memref=arr, indices=idxs):
                arr_name = self._get_variable_name_for(arr)
                idx_args = ", ".join(
                    map(self._get_variable_name_for, idxs))
                val_name = self._get_variable_name_for(val)
                self.print(f"{arr_name}[{idx_args}] = {val_name};")
            case memref.Load(memref=arr, indices=idxs, res=res):
                arr_name = self._get_variable_name_for(arr)
                idx_args = ", ".join(
                    map(self._get_variable_name_for, idxs))
                # Use the array access syntax instead of cipying the value out
                self.variables[res] = f"({arr_name}[{idx_args}])"
            case scf.Yield(arguments=args):
                pass
            case anyop:
                self.print(f"unknown op {anyop}", prefix="//")

    def print_module(self, mod: csl.ModuleOp):
        self.print_block(mod.body.block)

    def print_divider(self):
        self.print(self.DIVIDER)

    def descend(self) -> CslPrintContext:
        """
        Get a sub-context for descending into nested structures.

        Variables defined outside are valid inside, but inside varaibles will be
        available outside.
        """
        return CslPrintContext(
            output=self.output,
            variables=self.variables.copy(),
            _counter=self._counter,
            _prefix=self._prefix + (" " * self._INDENT_SIZE),
        )


def _get_layout_program(ops: BlockOps) -> tuple[csl.ModuleOp, csl.ModuleOp]:
    ops_list = list(ops)
    assert all(isinstance(mod, csl.ModuleOp) for mod in ops_list), \
        "Expected all top level ops to be csl.module"
    assert (len(ops_list) == 2), "Expected exactly two top level modules"
    ops_list = cast(list[csl.ModuleOp], ops_list)
    prog = next(filter(lambda mod: mod.kind.data == csl.ModuleKind.PROGRAM,
                       ops_list), None)
    layout = next(filter(lambda mod: mod.kind.data == csl.ModuleKind.LAYOUT,
                         ops_list), None)
    assert prog is not None and layout is not None, \
        "Expected exactly 1 program and exactly 1 layout module"

    return layout, prog


def print_to_csl(prog: ModuleOp, output: IO[str]):
    """
    Takes a module op and prints it to the given output stream.
    """
    ctx = CslPrintContext(output)
    layout, program = _get_layout_program(prog.body.block.ops)
    ctx.print_module(program)
    ctx.print_divider()
    ctx.print_module(layout)
