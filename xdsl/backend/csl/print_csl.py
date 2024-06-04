from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, Literal, cast

from xdsl.dialects import arith, csl, memref, scf
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
    TypeAttribute,
    UnitAttr,
)
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.irdl import Operand


@dataclass
class CslPrintContext:
    _INDEX = "i32"
    _INDENT = "  "
    DIVIDER = "// >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<< //"
    output: IO[str]

    variables: dict[SSAValue, str] = field(default_factory=dict)

    _counter: int = field(default=0)

    _prefix: str = field(default="")
    _symbols_to_export: dict[str, tuple[TypeAttribute, bool | None]] = field(
        default_factory=dict
    )

    def print(self, text: str, prefix: str = "", end: str = "\n"):
        """
        Print `text` line by line, prefixed by self._prefix and prefix.
        """
        for l in text.split("\n"):
            print(self._prefix + prefix + l, file=self.output, end=end)

    def _print_task_or_fn(
        self,
        kind: Literal["fn", "task"],
        name: StringAttr,
        bdy: Region,
        ftyp: FunctionType,
    ):
        """
        Shared printing logic for printing tasks and functions.
        """
        args = ", ".join(
            f"{self._get_variable_name_for(arg)} : {self.mlir_type_to_csl_type(arg.type)}"
            for arg in bdy.block.args
        )
        ret = (
            "void"
            if len(ftyp.outputs) == 0
            else self.mlir_type_to_csl_type(ftyp.outputs.data[0])
        )
        signature = f"\n{kind} {name.data}({args}) {ret}"
        with self.descend(signature) as inner:
            inner.print_block(bdy.block)

    def _wrap_task_id(self, kind: csl.TaskKind, id: int) -> str:
        """
        When using `@get_<kind>_tadk_id`, data task IDs have to be wrapped in
        `@get_color`. Local and control task IDs  just get passed directly.

        Returns wrapped ID as a string.
        """
        if kind == csl.TaskKind.DATA:
            return f"@get_color({id})"
        return str(id)

    def _print_bind_task(
        self, name: str, kind: csl.TaskKind, id: csl.ColorIdAttr | None
    ):
        """
        Generate a call to `@bind_<kind>_task` if task ID was specified as a
        property of the task. Otherwise we assume binding will be done at runtime.
        """
        if id is None:
            return
        with self.descend("comptime") as inner:
            inner.print(
                f"@bind_{kind.value}_task({name}, @get_{kind.value}_task_id({self._wrap_task_id(kind, id.value.data)}));"
            )

    def _memref_global_init(self, init: Attribute, type: str) -> str:
        """
        Generate an initialisation expression (@constants) for global arrays.
        Expects the memref.global initial_value property.
        """
        match init:
            case UnitAttr():
                return ""
            case DenseIntOrFPElementsAttr():
                data = init.data.data
                assert (
                    len(data) == 1
                ), f"Memref global initialiser has to have 1 value, got {len(data)}"
                return f" = @constants({type}, {self.attribute_value_to_str(data[0])})"
            case other:
                return f"<unknown memref.global init type {other}>"

    def _print_binop(self, lhs: Operand, rhs: Operand, res: SSAValue, op: str):
        """
        Prints statement of the form `res = lhs op rhs;`

        Used to print various binary operations.
        """
        name_lhs = self._get_variable_name_for(lhs)
        name_rhs = self._get_variable_name_for(rhs)
        self.print(f"{self._var_use(res)} = {name_lhs} {op} {name_rhs};")

    def _var_use(
        self, val: SSAValue, intro: Literal["const"] | Literal["var"] = "const"
    ):
        """
        Automates delcaration and use of variables.

        If a variable has not been declared (i.e. it's associated ssa value is
        not in self.variables), declare it with an approptiate name and type.
        Otherwise, just use the variable name.

        Optionally, introducer can be specified as var or const (const by default).
        """
        if val in self.variables:
            return f"{self._get_variable_name_for(val)}"
        else:
            return f"{intro} {self._get_variable_name_for(val)} : {self.mlir_type_to_csl_type(val.type)}"

    def _export_sym_constness(self, ty: FunctionType | csl.PtrType) -> bool | None:
        """
        Derive host-mutability for symbol exporting from MLIR type.

        When exporting symbols we have to specify if they can be modified by the
        host (true for mutable, false for immutable).

        This is only true for pointer types, function types are always immutable
        so their mutability cannot be specified (it's a compiler error to do so).
        We represent this by returning None.
        """
        if isinstance(ty, FunctionType):
            return None
        match ty.constness.data:
            case csl.PtrConst.VAR:
                return True
            case csl.PtrConst.CONST:
                return False

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
        - function: fn(i32) f16
        - color
        - comptime_struct
        - imported_module
        - type
        - comptime_string

        This method supports all of these except type and comptime_string
        """
        match type_attr:
            case csl.ComptimeStructType():
                return "comptime_struct"
            case csl.ImportedModuleType():
                return "imported_module"
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
                return f"i{width}"
            case MemRefType():
                t: ContainerType[TypeAttribute] = type_attr
                shape = ", ".join(str(s) for s in t.get_shape())
                type = self.mlir_type_to_csl_type(t.get_element_type())
                return f"[{shape}]{type}"
            case csl.PtrType(type=ty, kind=kind, constness=const):
                match kind.data:
                    case csl.PtrKind.SINGLE:
                        sym = "*"
                    case csl.PtrKind.MANY:
                        sym = "[*]"
                match const.data:
                    case csl.PtrConst.CONST:
                        mut = "const "
                    case csl.PtrConst.VAR:
                        mut = ""
                ty = self.mlir_type_to_csl_type(ty)
                return f"{sym}{mut}{ty}"
            case FunctionType(inputs=inp, outputs=out) if len(out) <= 1:
                args = map(self.mlir_type_to_csl_type, inp)
                ret = self.mlir_type_to_csl_type(out.data[0]) if len(out) else "void"
                return f"fn({', '.join(args)}) {ret}"
            case csl.ColorType():
                return "color"
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
            case IntegerAttr(
                value=val, type=IntegerType(width=IntAttr(data=width))
            ) if width == 1:
                return str(bool(val.data)).lower()
            case IntegerAttr(value=val):
                return str(val.data)
            case FloatAttr(value=val):
                return str(val.data)
            case StringAttr() as s:
                return f'"{s.data}"'
            case _:
                return f"<!unknown value {attr}>"

    def print_block(self, body: Block):
        """
        Walks over a block and prints every operation in the block.
        """
        for op in body.ops:
            match op:
                case arith.Constant(value=v, result=r):
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

                    res_type = self.mlir_type_to_csl_type(res.type)

                    self.print(
                        f'const {name} : {res_type} = @import_module("{module.data}"{params_str});'
                    )
                case csl.MemberCallOp(
                    struct=struct, field=field, args=args, result=res
                ):
                    args = ", ".join(self._get_variable_name_for(arg) for arg in args)
                    struct_var = self._get_variable_name_for(struct)

                    var = f"{self._var_use(res)} = " if res is not None else ""
                    self.print(f"{var}{struct_var}.{field.data}({args});")
                case csl.CallOp(callee=callee, args=args, result=res):
                    args = ", ".join(map(self._get_variable_name_for, args))
                    var = f"{self._var_use(res)} = " if res is not None else ""
                    self.print(f"{var}{callee.string_value()}({args});")
                case csl.MemberAccessOp(struct=struct, field=field, result=res):
                    struct_var = self._get_variable_name_for(struct)
                    self.print(f"{self._var_use(res)} = {struct_var}.{field.data};")
                case csl.TaskOp(
                    sym_name=name, body=bdy, function_type=ftyp, kind=kind, id=id
                ):
                    self._print_task_or_fn("task", name, bdy, ftyp)
                    self._print_bind_task(name.data, kind.data, id)
                case csl.FuncOp(sym_name=name, body=bdy, function_type=ftyp):
                    self._print_task_or_fn("fn", name, bdy, ftyp)
                case csl.ReturnOp(ret_val=None):
                    self.print("return;")
                case csl.ReturnOp(ret_val=val) if val is not None:
                    self.print(f"return {self._get_variable_name_for(val)};")
                case scf.For(
                    lb=lower, ub=upper, step=stp, body=bdy, res=results, iter_args=iters
                ):
                    idx, *args = bdy.block.args
                    # declare for loop iterators as mutable variables and match their names to for result names
                    for result, iter, arg in zip(results, iters, args):
                        iter_name = self._get_variable_name_for(iter)
                        self.print(f"{self._var_use(result, 'var')} = {iter_name};")
                        self.variables[arg] = self.variables[result]
                    # Search for all yield operations and match yield argument names to for argument names
                    for op in bdy.block.ops:
                        if not isinstance(op, scf.Yield):
                            continue
                        for yield_arg, arg in zip(op.arguments, args):
                            self.variables[yield_arg] = self.variables[arg]

                    idx_type = self.mlir_type_to_csl_type(idx.type)
                    lower_name, upper_name, stp_name, idx_name = map(
                        self._get_variable_name_for, (lower, upper, stp, idx)
                    )
                    loop_definition = f"\nfor(@range({idx_type}, {lower_name}, {upper_name}, {stp_name})) |{idx_name}|"
                    with self.descend(loop_definition) as inner:
                        inner.print_block(bdy.block)
                case scf.Yield():
                    pass
                case (
                    arith.IndexCastOp(input=inp, result=res)
                    | arith.SIToFPOp(input=inp, result=res)
                    | arith.FPToSIOp(input=inp, result=res)
                    | arith.ExtFOp(input=inp, result=res)
                    | arith.TruncFOp(input=inp, result=res)
                    | arith.TruncIOp(input=inp, result=res)
                    | arith.ExtSIOp(input=inp, result=res)
                    | arith.ExtUIOp(input=inp, result=res)
                ):
                    name_in = self._get_variable_name_for(inp)
                    type_out = self.mlir_type_to_csl_type(res.type)
                    self.print(f"{self._var_use(res)} = @as({type_out}, {name_in});")
                case arith.Muli(lhs=lhs, rhs=rhs, result=res) | arith.Mulf(
                    lhs=lhs, rhs=rhs, result=res
                ):
                    self._print_binop(lhs, rhs, res, "*")
                case arith.Addi(lhs=lhs, rhs=rhs, result=res) | arith.Addf(
                    lhs=lhs, rhs=rhs, result=res
                ):
                    self._print_binop(lhs, rhs, res, "+")
                case memref.Global(
                    sym_name=name, type=ty, initial_value=init, constant=const
                ):
                    name = name.data
                    ty = self.mlir_type_to_csl_type(ty)
                    init = self._memref_global_init(init, ty)
                    var = "var" if const is None else "const"
                    self.print(f"{var} {name} : {ty}{init};")
                case memref.GetGlobal(name_=name, memref=res):
                    # We print the array definition when the global is defined
                    self.variables[res] = name.string_value()
                case memref.Store(value=val, memref=arr, indices=idxs):
                    arr_name = self._get_variable_name_for(arr)
                    idx_args = ", ".join(map(self._get_variable_name_for, idxs))
                    val_name = self._get_variable_name_for(val)
                    self.print(f"{arr_name}[{idx_args}] = {val_name};")
                case memref.Load(memref=arr, indices=idxs, res=res):
                    arr_name = self._get_variable_name_for(arr)
                    idx_args = ", ".join(map(self._get_variable_name_for, idxs))
                    # Use the array access syntax instead of copying the value out
                    self.variables[res] = f"({arr_name}[{idx_args}])"
                case csl.AddressOfOp(value=val, res=res):
                    val_name = self._get_variable_name_for(val)
                    ty = cast(csl.PtrType, res.type)
                    use = self._var_use(res, ty.constness.data.value)
                    self.print(f"{use} = &{val_name};")
                case csl.SymbolExportOp(value=val, type=ty) as exp:
                    name = exp.get_name()
                    q_name = f'"{name}"'
                    self._symbols_to_export[name] = (ty, self._export_sym_constness(ty))
                    ty = self.attribute_value_to_str(ty)
                    if val is not None:
                        export_val = self._get_variable_name_for(val)
                    else:
                        # Use symbol ref name if operand not provided
                        export_val = name
                    with self.descend("comptime") as inner:
                        inner.print(f"@export_symbol({export_val}, {q_name});")
                case csl.LayoutOp(body=bdy):
                    with self.descend("layout") as inner:
                        inner.print_block(bdy.block)
                        for name, val in inner._symbols_to_export.items():
                            ty = inner.mlir_type_to_csl_type(val[0])
                            # If specified, get mutability as true/false from python bool
                            mut = str(val[1]).lower() if val[1] is not None else ""
                            inner.print(f'@export_name("{name}", {ty}, {mut});')
                case csl.ParamOp(init_value=init, param_name=name, res=res):
                    if init is None:
                        init = ""
                    else:
                        init = f" = { self.attribute_value_to_str(init)}"
                    ty = self.mlir_type_to_csl_type(res.type)
                    self.print(f"param {name.data} : {ty}{init};")
                case csl.ConstStructOp(
                    items=items, ssa_fields=fields, ssa_values=values, res=res
                ):
                    items = items or DictionaryAttr({})
                    fields = fields or ArrayAttr([])
                    # First print the fields defined by attributes
                    self.print(f"{self._var_use(res)} = .{{")
                    for k, v in items.data.items():
                        v = self.attribute_value_to_str(v)
                        self.print(f".{k} = {v},", prefix=self._INDENT)
                    # Then the fields defined by operands, with their corresponding names
                    for k, v in zip(fields.data, values):
                        v = self._get_variable_name_for(v)
                        self.print(f".{k.data} = {v},", prefix=self._INDENT)
                    self.print("};")
                case csl.SetTileCodeOp(
                    file=file, x_coord=x_coord, y_coord=y_coord, params=params
                ):
                    file = self.attribute_value_to_str(file)
                    x = self._get_variable_name_for(x_coord)
                    y = self._get_variable_name_for(y_coord)
                    params = self._get_variable_name_for(params) if params else ""
                    self.print(f"@set_tile_code({x}, {y}, {file}, {params});")
                case csl.SetRectangleOp(x_dim=x_dim, y_dim=y_dim):
                    x = self._get_variable_name_for(x_dim)
                    y = self._get_variable_name_for(y_dim)
                    self.print(f"@set_rectangle({x}, {y});")
                case csl.GetColorOp(id=id, res=res):
                    id = self.attribute_value_to_str(id)
                    self.print(f"{self._var_use(res)} = @get_color({id});")
                case csl.RpcOp(id=id):
                    id = self._get_variable_name_for(id)
                    with self.descend("comptime") as inner:
                        inner.print(f"@rpc(@get_data_task_id({id}));")
                case anyop:
                    self.print(f"unknown op {anyop}", prefix="//")

    @contextmanager
    def descend(self, block_start: str = ""):
        """
        Get a sub-context for descending into nested structures.

        Variables defined outside are valid inside, but inside varaibles will be
        available outside.

        The code printed in this context will be surrounded by curly braces and
        can optionally start with a `block_start` statement (e.g. function
        siganture or the `comptime` keyword).

        To be used in a `with` statement like so:
        ```
        with self.descend() as inner_context:
            inner_context.print()
        ```

        NOTE: `_symbols_to_export` is passed as a reference, so the sub-context
        could in theory modify the parent's list of exported symbols, in
        practice this should not happen as `SymbolExportOp` has been verified to
        only be present at module scope.
        """
        if block_start != "":
            block_start = f"{block_start} "
        self.print(f"{block_start}{{")
        yield CslPrintContext(
            output=self.output,
            variables=self.variables.copy(),
            _symbols_to_export=self._symbols_to_export,
            _counter=self._counter,
            _prefix=self._prefix + self._INDENT,
        )
        self.print("}")


def _get_layout_program(module: ModuleOp) -> tuple[csl.CslModuleOp, csl.CslModuleOp]:
    """
    Get the layout and program `csl.module`s from the top level `builtin.module`.

    Makes sure there is exactly 1 layout and 1 program `csl.module`.

    Returns layout first, then program.
    """
    ops_list = list(module.body.block.ops)
    assert all(
        isinstance(mod, csl.CslModuleOp) for mod in ops_list
    ), "Expected all top level ops to be csl.module"
    # We have asserted that this is true above
    ops_list = cast(list[csl.CslModuleOp], ops_list)
    assert len(ops_list) == 2, "Expected exactly two top level modules"
    # This allows program and layout to be scpecified in any order
    prog = next(
        filter(lambda mod: mod.kind.data == csl.ModuleKind.PROGRAM, ops_list), None
    )
    layout = next(
        filter(lambda mod: mod.kind.data == csl.ModuleKind.LAYOUT, ops_list), None
    )
    assert prog is not None, "Expected exactly 1 program module"
    assert layout is not None, "Expected exactly 1 layout module"
    return layout, prog


def print_to_csl(prog: ModuleOp, output: IO[str]):
    """
    Takes a module op and prints it to the given output stream.
    """
    ctx = CslPrintContext(output)
    layout, program = _get_layout_program(prog)
    ctx.print_block(program.body.block)
    ctx.print(ctx.DIVIDER)
    ctx.print_block(layout.body.block)
