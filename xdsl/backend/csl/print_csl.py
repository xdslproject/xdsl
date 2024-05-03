from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, cast
from contextlib import contextmanager

from xdsl.dialects import arith, csl, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    SignednessAttr,
    StringAttr,
    TensorType,
)
from xdsl.ir import Attribute, Block, SSAValue, Region
from xdsl.ir.core import BlockOps, TypeAttribute


@dataclass
class CslPrintContext:
    _INDENT_SIZE = 2
    DIVIDER = "// >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<< //"
    output: IO[str]

    variables: dict[SSAValue, str] = field(default_factory=dict)

    _counter: int = field(default=0)

    _prefix: str = field(default="")

    _symbols_to_export: dict[str, tuple[TypeAttribute, bool]] \
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
            case IntegerType(
                width=IntAttr(data=width),
                signedness=SignednessAttr(data=Signedness.UNSIGNED),
            ):
                return f"u{width}"
            case IntegerType(width=IntAttr(data=width)):
                return "bool" if width == 1 else f"i{width}"
            case FunctionType(inputs=inp, outputs=out) if len(out) == 1:
                args = map(self.mlir_type_to_csl_type, inp)
                ret = self.mlir_type_to_csl_type(out.data[0])
                return f"fn({', '.join(args)}) {ret}"
            case TensorType():
                t: TensorType[TypeAttribute] = type_attr
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

    def _is_mutable(self, const: csl.PtrConst) -> bool:
        match const:
            case csl.PtrConst.MUT: return True
            case csl.PtrConst.CONST: return False

    def print_block(self, body: Block):
        """
        Walks over a block and prints every operation in the block.
        """
        for op in body.ops:
            match op:
                case arith.Constant(value=v, result=r) \
                        | csl.ConstStrOp(string=v, res=r)\
                        | csl.ConstTypeOp(type=v, res=r):
                    # v is an attribute that "carries a value", e.g. an IntegerAttr or FloatAttr

                    # convert the attributes type to a csl type:
                    type_name = self.attribute_type_to_str(v)
                    # convert the carried value to a csl value
                    value_str = self.attribute_value_to_str(v)

                    # emit a constant instantiation:
                    self.print(
                        f"const {self._get_variable_name_for(r)} : {type_name} = {
                            value_str};"
                    )
                case csl.ImportModuleConstOp(module=module, params=params, result=res):
                    name = self._get_variable_name_for(res)

                    params_str = ""
                    if params is not None:
                        params_str = f", {self._get_variable_name_for(params)}"

                    self.print(
                        f'const {name} : imported_module = @import_module("{
                            module.data}"{params_str});'
                    )
                case csl.MemberCallOp(field=callee, args=args, result=res) \
                        | csl.CallOp(callee=callee, args=args, result=res) as call:
                    args = ", ".join(self._get_variable_name_for(arg)
                                     for arg in args)
                    if struct := getattr(call, "struct", None):
                        struct_str = f"{self._get_variable_name_for(struct)}."
                    else:
                        struct_str = ""

                    text = ""
                    if res is not None:
                        name = self._get_variable_name_for(res)
                        text += (
                            f"const {name} : {
                                self.mlir_type_to_csl_type(res.type)} = "
                        )

                    self.print(f"{text}{struct_str}{callee.data}({args});")
                case csl.MemberAccessOp(struct=struct, field=field, result=res):
                    name = self._get_variable_name_for(res)
                    struct_var = self._get_variable_name_for(struct)
                    self.print(
                        f"const {name} : {self.mlir_type_to_csl_type(res.type)} = {
                            struct_var}.{field.data};"
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
                case scf.For(lb=lower, ub=upper, step=stp, body=bdy):
                    idx, *_ = bdy.block.args
                    self.print(
                        f"\nfor(@range({self.mlir_type_to_csl_type(idx.type)}, {self._get_variable_name_for(lower)}, {
                            self._get_variable_name_for(upper)}, {self._get_variable_name_for(stp)})) |{self._get_variable_name_for(idx)}| {{"
                    )
                    self.descend().print_block(bdy.block)
                    self.print("}")
                case csl.LayoutOp(body=bdy):
                    with self._in_block("layout"):
                        self.print_block(bdy.block)
                        for name, val in self._symbols_to_export.items():
                            ty = self.attribute_value_to_str(val[0])
                            mut = str(val[1]).lower()
                            self.print(f"@export_name({name}, {ty}, {mut});")
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
                case csl.SymbolExportOp(value=val, var_name=name, type=ty):
                    name = self.attribute_value_to_str(name)
                    self._symbols_to_export[name] = (
                        ty,
                        self._is_mutable(ty.constness.data)
                    )
                    ty = self.attribute_value_to_str(ty)
                    val = self._get_variable_name_for(val)
                    with self._in_block("comptime"):
                        self.print(f"@export_symbol({val}, {name});")
                case csl.GetColorOp(id=id, res=res):
                    id = self.attribute_value_to_str(id)
                    var = self._get_variable_name_for(res)
                    color_t = self.mlir_type_to_csl_type(res.type)
                    self.print(f"const {var} : {color_t} = @get_color({id});")
                case csl.ConstStructOp(items=items, ssa_fields=fields, ssa_values=values, res=res):
                    var = self._get_variable_name_for(res)
                    struct_t = self.mlir_type_to_csl_type(res.type)
                    items = items or DictionaryAttr({})
                    fields = fields or ArrayAttr([])
                    self.print(f"const {var} : {struct_t} = .{{")
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
                    res_name = self._get_variable_name_for(res)
                    res_type = cast(csl.PtrType, res.type)
                    const = self._ptr_kind_to_introducer(
                        res_type.constness.data)
                    res_type = self.mlir_type_to_csl_type(res.type)
                    self.print(
                        f"{const} {res_name} : {res_type} = &{val_name};")
                case csl.ArrayOp(init_value=init, type=ty, res=res):
                    type = self.mlir_type_to_csl_type(ty)
                    if init is not None:
                        val = self.attribute_value_to_str(init)
                        init = f" = @constants({type}, {val})"
                    else:
                        init = ""
                    name = self._get_variable_name_for(res)
                    self.print(f"var {name} : {type}{init};")
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
