from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, Literal, cast

from xdsl.dialects import arith, csl, memref, scf
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ArrayAttr,
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
    i1,
)
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.irdl import Operand
from xdsl.traits import is_side_effect_free
from xdsl.utils.comparisons import to_unsigned
from xdsl.utils.hints import isa

_CSL_KW_SET = {
    "align",
    "and",
    "bool",
    "break",
    "comptime_float",
    "comptime_int",
    "comptime_string",
    "comptime_struct",
    "const",
    "continue",
    "else",
    "export",
    "extern",
    "f16",
    "f32",
    "false",
    "fn",
    "for",
    "i16",
    "i32",
    "i64",
    "i8",
    "if",
    "linkname",
    "linksection",
    "or",
    "param",
    "return",
    "switch",
    "task",
    "true",
    "u16",
    "u32",
    "u64",
    "u8",
    "var",
    "void",
    "while",
}
"""
The set of CSL language keywords. These should not be used as variable names.

There is no official list of all reserved keywords in CSL, this list was
compiled using the keywords found [here](https://sdk.cerebras.net/csl/language/syntax)
and should be expanded as needed.
"""


@dataclass
class CslPrintContext:
    _INDEX = "i32"
    _INDENT = "  "
    output: IO[str]

    variables: dict[SSAValue, str] = field(default_factory=dict[SSAValue, str])

    _counter: int = field(default=0)

    _prefix: str = field(default="")
    _symbols_to_export: dict[str, tuple[TypeAttribute, bool | None]] = field(
        default_factory=dict[str, tuple[TypeAttribute, bool | None]]
    )

    _binops: dict[str, str] = field(default_factory=dict[str, str])
    """
    Maps operation name => operand for binary operands
    """

    _cmp_ops: dict[str, dict[str, str | None]] = field(
        default_factory=dict[str, dict[str, str | None]]
    )

    def register_binops(self):
        self._binops.update(
            {
                arith.AddfOp.name: "+",
                arith.AddiOp.name: "+",
                arith.MulfOp.name: "*",
                arith.MuliOp.name: "*",
                arith.DivfOp.name: "/",
                arith.DivSIOp.name: "/",
                arith.DivUIOp.name: "/",
                arith.SubfOp.name: "-",
                arith.SubiOp.name: "-",
                arith.RemSIOp.name: "%",
                arith.RemUIOp.name: "%",
                arith.ShLIOp.name: "<<",
                arith.AndIOp.name: "&",
                arith.OrIOp.name: "|",
            }
        )
        self._cmp_ops.update(
            {
                arith.CmpiOp.name: {
                    "eq": "==",
                    "ne": "!=",
                    "slt": "<",
                    "sle": "<=",
                    "sgt": ">",
                    "sge": ">=",
                    "ult": "<",
                    "ule": "<=",
                    "ugt": ">",
                    "uge": ">=",
                },
                arith.CmpfOp.name: {
                    "false": None,
                    "oeq": "==",
                    "ogt": ">",
                    "oge": ">=",
                    "olt": "<",
                    "ole": "<=",
                    "one": "!=",
                    "ord": None,
                    "ueq": "==",
                    "ugt": ">",
                    "uge": ">=",
                    "ult": "<",
                    "ule": "<=",
                    "une": "!=",
                    "uno": None,
                    "true": None,
                },
            }
        )

    def _cmp_value_expr(self, op: arith.CmpiOp | arith.CmpfOp):
        pred = op.predicate.value.data
        str_pred = {
            arith.CmpiOp.name: arith.CMPI_COMPARISON_OPERATIONS,
            arith.CmpfOp.name: arith.CMPF_COMPARISON_OPERATIONS,
        }[op.name][pred]
        lhs_name = self._get_variable_name_for(op.lhs)
        rhs_name = self._get_variable_name_for(op.rhs)

        if sym := self._cmp_ops[op.name][str_pred]:
            return f"{lhs_name}  {sym}  {rhs_name}"
        match str_pred:
            case "true" | "false":
                return str_pred
            case "ord" | "uno":
                raise RuntimeError(f"{str_pred}: comparison not supported")
            case unknown:
                raise RuntimeError(f"Unknown predicate {unknown}")

    def _binop_value_expr(self, op: Operation):
        assert len(op.operands) == 2, "binops must have exactly two operands"
        assert op.name in self._binops, "unknown binop"
        a, b = map(self._get_variable_name_for, op.operands)

        return f"{a} {self._binops[op.name]} {b}"

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
            unsigned_id = to_unsigned(id.value.data, id.type.width.data)
            inner.print(
                f"@bind_{kind.value}_task({name}, @get_{kind.value}_task_id({self._wrap_task_id(kind, unsigned_id)}));"
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
                data = init.get_attrs()
                assert len(data) == 1, (
                    f"MemRef global initialiser has to have 1 value, got {len(data)}"
                )
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

    def _var_use(self, val: SSAValue, intro: Literal["const", "var"] = "const"):
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

        taken_names = set(self.variables.values()) | _CSL_KW_SET

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

    def _memref_type_to_string(self, val: SSAValue):
        """
        Generate the string representing the type of a memref.

        We need to get the SSAValue passed here, as for unknown sizes, we need to put the
        variable name in the type.

        For every unknown size (DYNAMIC_INDEX) in the shape, we look up the corresponding operand to the value
        that created the memref (e.g. memref.alloc, csl.constants, ...)
        """
        type = val.type
        assert isa(type, MemRefType)
        assert isinstance(val, OpResult), (
            "The value provided to _memref_type_to_string must be an op result"
        )
        dims: list[str] = []
        idx = 0
        for dim in type.get_shape():
            if dim == DYNAMIC_INDEX:
                dims.append(self.variables[val.owner.operands[idx]])
                idx += 1
            else:
                dims.append(str(dim))
        dims_str = ",".join(dims)
        return f"[{dims_str}]{self.mlir_type_to_csl_type(type.element_type)}"

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
            case IntegerType(width=IntAttr(1)):
                return "bool"
            case IntegerType(
                signedness=SignednessAttr(data=Signedness.UNSIGNED),
            ):
                return f"u{cast(IntegerType, type_attr).width.data}"
            case IntegerType():
                return f"i{cast(IntegerType, type_attr).width.data}"
            case MemRefType(element_type=Attribute() as elem_t, shape=shape):
                if any(dim.data == DYNAMIC_INDEX for dim in shape):
                    raise ValueError(
                        "Can't print memrefs using mlir_type_to_csl_type if they have dynamic sizes. "
                        "Use _memref_type_to_string instead"
                    )
                shape = ", ".join(str(s.data) for s in shape)
                type = self.mlir_type_to_csl_type(elem_t)
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
            case csl.DirectionType():
                return "direction"
            case FunctionType(inputs=inp, outputs=out) if len(out) <= 1:
                args = map(self.mlir_type_to_csl_type, inp)
                ret = self.mlir_type_to_csl_type(out.data[0]) if len(out) else "void"
                return f"fn({', '.join(args)}) {ret}"
            case csl.ColorType():
                return "color"
            case csl.DsdType() as dsd:
                return dsd.data
            case csl.VarType() as v:
                return self.mlir_type_to_csl_type(v.get_element_type())
            case _:
                return f"<!unknown type {type_attr}>"

    def attribute_value_to_str(self, attr: Attribute) -> str:
        """
        Takes a value-carrying attribute (IntegerAttr, FloatAttr, etc.)
        and converts it to a csl expression representing that value literal (0, 3.14, ...)
        """
        match attr:
            case IntAttr():
                return str(cast(IntAttr[int], attr).data)
            case IntegerAttr(value=val, type=IntegerType(width=IntAttr(data=1))):
                return str(bool(val.data)).lower()
            case IntegerAttr(value=val):
                return str(val.data)
            case FloatAttr(value=val) if val.data == 0:
                return "0.0"
            case FloatAttr(value=val):
                return str(val.data)
            case StringAttr() as s:
                return f'"{s.data}"'
            case DenseIntOrFPElementsAttr():
                return f"{self.mlir_type_to_csl_type(attr.get_type())} {{ {', '.join(self.attribute_value_to_str(a) for a in attr.iter_attrs())} }}"  # noqa: E501
            case _:
                return f"<!unknown value {attr}>"

    def _can_promote_result_to_inline_expr(self, var: OpResult):
        """
        Check if a result can be promoted to an immediate. This is only the case if:

        - The variable is not already assigned to a variable name (then we expect that results be assigned there)
          This happens for example for loop carried variables.
        - The operation itself is side effect free (e.g. not a load/store)
        - At least one result is used somewhere (if there are no uses of it, promoting to immediate would erase it)
        - The variable that would be created isn't used by an AddressOf operation.
        """
        return (
            var not in self.variables
            and is_side_effect_free(var.owner)
            and any(res.uses for res in var.owner.results)
            and not any(
                isinstance(use.operation, csl.AddressOfOp)
                for res in var.owner.results
                for use in res.uses
            )
        )

    def _print_or_promote_to_inline_expr(
        self, var: OpResult, value_expr: str, brackets: bool = False
    ):
        """
        Given an SSA value (op result) and a string representing its value.

        Check that the result can be promoted to an expression, or if not
        assign it to a new variable.

        Optionally adds brackets around the value when promoting to expression.
        """
        # prevent exploding expression sizes
        # also check that the expression is safe to promote
        if len(value_expr) < 50 and self._can_promote_result_to_inline_expr(var):
            if brackets:
                value_expr = f"({value_expr})"
            self.variables[var] = value_expr
        else:
            self.print(f"{self._var_use(var)} = {value_expr};")

    def print_op(self, op: Operation):
        match op:
            case (
                arith.AndIOp(lhs=lhs, rhs=rhs, result=res)
                | arith.OrIOp(lhs=lhs, rhs=rhs, result=res)
            ) if res.type == i1:
                lhs_name = self._get_variable_name_for(lhs)
                rhs_name = self._get_variable_name_for(rhs)
                self._print_or_promote_to_inline_expr(
                    res,
                    f"{lhs_name} {'or' if isa(op, arith.OrIOp) else 'and'} {rhs_name}",
                    brackets=True,
                )
            # handle all binary ops at once:
            case Operation() if op.name in self._binops:
                self._print_or_promote_to_inline_expr(
                    op.results[0], self._binop_value_expr(op), brackets=True
                )
            case arith.ConstantOp(value=v, result=r):
                self._print_or_promote_to_inline_expr(r, self.attribute_value_to_str(v))
            case csl.ImportModuleConstOp(module=module, params=params, result=res):
                name = self._get_variable_name_for(res)

                params_str = ""
                if params is not None:
                    params_str = f", {self._get_variable_name_for(params)}"

                res_type = self.mlir_type_to_csl_type(res.type)

                self.print(
                    f'const {name} : {res_type} = @import_module("{module.data}"{params_str});'
                )
            case csl.MemberCallOp(struct=struct, field=field, args=args, result=res):
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
                self._print_or_promote_to_inline_expr(res, f"{struct_var}.{field.data}")
            case csl.TaskOp(
                sym_name=name, body=bdy, function_type=ftyp, kind=kind, id=id
            ):
                self._print_task_or_fn("task", name, bdy, ftyp)
                self._print_bind_task(name.data, kind.data, id)
            case csl.FuncOp(sym_name=name, body=bdy, function_type=ftyp):
                self._print_task_or_fn("fn", name, bdy, ftyp)
            case csl.ActivateOp(id=id, kind=kind):
                id = self.attribute_value_to_str(id)
                self.print(f"@activate(@get_{kind.data.value}_task_id({id}));")
            case csl.ReturnOp(ret_val=None):
                self.print("return;")
            case csl.ReturnOp(ret_val=val) if val is not None:
                self.print(f"return {self._get_variable_name_for(val)};")
            case scf.IfOp(
                cond=cond,
                output=outputs,
                true_region=true_region,
                false_region=false_region,
            ):
                for o in outputs:
                    self.print(f"{self._var_use(o, 'var')};")
                # Search for all yield operations and match yield argument names to for argument names
                for blk in [true_region, false_region]:
                    if isinstance(blk.block.last_op, scf.YieldOp):
                        for yield_arg, o in zip(blk.block.last_op.arguments, outputs):
                            self.variables[yield_arg] = self.variables[o]
                with self.descend(f"if ({self._get_variable_name_for(cond)})") as inner:
                    inner.print_block(true_region.block)
                if false_region:
                    if len(outputs) > 0 or not (
                        len(false_region.block.ops) == 1
                        and isinstance(false_region.block.first_op, scf.YieldOp)
                    ):
                        with self.descend("else") as inner:
                            inner.print_block(false_region.block)
            case scf.ForOp(
                lb=lower, ub=upper, step=stp, body=bdy, res=results, iter_args=iters
            ):
                idx, *args = bdy.block.args
                # declare for loop iterators as mutable variables and match their names to for result names
                for result, iter, arg in zip(results, iters, args):
                    iter_name = self._get_variable_name_for(iter)
                    self.print(f"{self._var_use(result, 'var')} = {iter_name};")
                    self.variables[arg] = self.variables[result]
                # Search for all yield operations and match yield argument names to for argument names
                if isinstance(bdy.block.last_op, scf.YieldOp):
                    for yield_arg, arg in zip(bdy.block.last_op.arguments, args):
                        self.variables[yield_arg] = self.variables[arg]

                idx_type = self.mlir_type_to_csl_type(idx.type)
                lower_name, upper_name, stp_name, idx_name = map(
                    self._get_variable_name_for, (lower, upper, stp, idx)
                )
                loop_definition = f"\nfor(@range({idx_type}, {lower_name}, {upper_name}, {stp_name})) |{idx_name}|"
                with self.descend(loop_definition) as inner:
                    inner.print_block(bdy.block)
            case scf.YieldOp():
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
                | csl.SignednessCastOp(inp=inp, result=res)
            ):
                name_in = self._get_variable_name_for(inp)
                type_out = self.mlir_type_to_csl_type(res.type)
                value_str = f"@as({type_out}, {name_in})"
                self._print_or_promote_to_inline_expr(res, value_str)
            case arith.CmpiOp(result=res) | arith.CmpfOp(result=res):
                self._print_or_promote_to_inline_expr(
                    res, self._cmp_value_expr(op), brackets=True
                )
            case arith.SelectOp(cond=cond, lhs=lhs, rhs=rhs, result=res):
                cond = self._get_variable_name_for(cond)
                lhs = self._get_variable_name_for(lhs)
                rhs = self._get_variable_name_for(rhs)
                if_str = f"if ({cond}) {lhs} else {rhs}"
                self._print_or_promote_to_inline_expr(res, if_str, brackets=True)
            case csl.ConcatStructOp(this_struct=a, another_struct=b, result=res):
                a_var = self._get_variable_name_for(a)
                b_var = self._get_variable_name_for(b)
                self._print_or_promote_to_inline_expr(
                    res, f"@concat_structs({a_var}, {b_var})"
                )
            case csl.ZerosOp(result=res, is_const=constness):
                type = self._memref_type_to_string(res)
                res_name = self._get_variable_name_for(res)
                kind = "const" if constness else "var"
                self.print(f"{kind} {res_name} : {type} = @zeros({type});")
            case csl.ConstantsOp(value=val, result=res, is_const=constness):
                type = self._memref_type_to_string(res)
                res_name = self._get_variable_name_for(res)
                kind = "const" if constness else "var"
                self.print(
                    f"{kind} {res_name} : {type} = @constants({type}, {self._var_use(val)});"
                )
            case memref.GlobalOp(
                sym_name=name, type=ty, initial_value=init, constant=const
            ):
                name = name.data
                ty = self.mlir_type_to_csl_type(ty)
                init = self._memref_global_init(init, ty)
                var = "var" if const is None else "const"
                self.print(f"{var} {name} : {ty}{init};")
            case memref.GetGlobalOp(name_=name, memref=res):
                # We print the array definition when the global is defined
                self.variables[res] = name.string_value()
            case memref.StoreOp(value=val, memref=arr, indices=idxs):
                arr_name = self._get_variable_name_for(arr)
                idx_args = ", ".join(map(self._get_variable_name_for, idxs))
                val_name = self._get_variable_name_for(val)
                self.print(f"{arr_name}[{idx_args}] = {val_name};")
            case memref.LoadOp(memref=arr, indices=idxs, res=res):
                arr_name = self._get_variable_name_for(arr)
                idx_args = ", ".join(map(self._get_variable_name_for, idxs))
                # Use the array access syntax instead of copying the value out
                self.variables[res] = f"({arr_name}[{idx_args}])"
            case csl.AddressOfOp(value=val, res=res):
                val_name = self._get_variable_name_for(val)
                use = self._var_use(res, res.type.constness.data.value)
                self.print(f"{use} = &{val_name};")

            case csl.AddressOfFnOp(fn_name=name, res=res):
                use = self._var_use(res, res.type.constness.data.value)
                self.print(f"{use} = &{name.string_value()};")
            case csl.DirectionOp(dir=d, res=res):
                self._print_or_promote_to_inline_expr(res, str.upper(d.data))
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
                    init = f" = {self._get_variable_name_for(init)}"
                ty = self.mlir_type_to_csl_type(res.type)
                self.variables[res] = name.data
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
                id = self._get_variable_name_for(id)
                self._print_or_promote_to_inline_expr(res, f"@get_color({id})")
            case csl.RpcOp(id=id):
                id = self._get_variable_name_for(id)
                with self.descend("comptime") as inner:
                    inner.print(f"@rpc(@get_data_task_id({id}));")
            case csl.GetMemDsdOp(
                base_addr=base_addr,
                tensor_access=tensor_access,
                sizes=sizes,
                result=result,
            ):
                sizes_str = ", ".join(
                    self._get_variable_name_for(size) for size in sizes
                )
                t_accesses = (
                    tensor_access.data
                    if tensor_access
                    else AffineMap.identity(len(sizes))
                )

                ind_vars = ["d" + str(i) for i in range(len(sizes))]
                ind_vars_str = ", ".join(ind_vars)
                accesses_str = ", ".join(str(expr) for expr in t_accesses.results)
                self.print(
                    f"{self._var_use(result)} = @get_dsd( {self.mlir_type_to_csl_type(result.type)}, .{{"
                )
                self.print(
                    f"  .tensor_access = | {ind_vars_str} | {{ {sizes_str} }} -> {self._var_use(base_addr)}[ {accesses_str} ]"
                )
                self.print("});")
            case csl.GetFabDsdOp(
                sizes=extent,
                fabric_color=fabric_color,
                queue_id=queue_id,
                control=control,
                wavelet_index_offset=wavelet_index_offset,
                result=result,
            ):
                self.print(
                    f"{self._var_use(result)} = @get_dsd({self.mlir_type_to_csl_type(result.type)}, .{{ "
                )
                self.print(f"  .extent = {self._get_variable_name_for(extent[0])},")
                q_type = (
                    "input"
                    if result.type == csl.DsdType(csl.DsdKind.fabin_dsd)
                    else "output"
                )
                self.print(
                    f"  .{q_type}_queue = @get_{q_type}_queue({queue_id.value.data}),"
                )
                self.print(f"  .fabric_color = {fabric_color},")
                if wavelet_index_offset is not None:
                    self.print(f"  .wavelet_index_offset = {wavelet_index_offset},")
                if control is not None:
                    self.print(f"  .control = {control},")
                self.print("}});")
            case csl.SetDsdBaseAddrOp(op=input_dsd, base_addr=base_addr, result=result):
                self.print(
                    f"{self._var_use(result)} = @set_dsd_base_addr({self._get_variable_name_for(input_dsd)}, {self._get_variable_name_for(base_addr)});"  # noqa: E501
                )
            case csl.IncrementDsdOffsetOp(
                op=input_dsd, offset=offset, elem_type=elem_type, result=result
            ):
                self.print(
                    f"{self._var_use(result)} = @increment_dsd_offset({self._get_variable_name_for(input_dsd)}, {self._get_variable_name_for(offset)}, {self.mlir_type_to_csl_type(elem_type)});"  # noqa: E501
                )
            case csl.SetDsdLengthOp(op=input_dsd, length=length, result=result):
                self.print(
                    f"{self._var_use(result)} = @set_dsd_length({self._get_variable_name_for(input_dsd)}, {self._get_variable_name_for(length)});"  # noqa: E501
                )
            case csl.SetDsdStrideOp(op=input_dsd, stride=stride, result=result):
                self.print(
                    f"{self._var_use(result)} = @set_dsd_stride({self._get_variable_name_for(input_dsd)}, {self._get_variable_name_for(stride)});"  # noqa: E501
                )
            case csl.BuiltinDsdOp(ops=ops):
                self.print(
                    f"@{op.name.removeprefix('csl.')}({', '.join(map(self._get_variable_name_for, ops))});"
                )
            case csl.VariableOp(default=default, res=res):
                var = self._var_use(res, "var")
                init_val = (
                    f" = {self.attribute_value_to_str(default)}"
                    if default is not None
                    else ""
                )
                self.print(f"{var}{init_val};")
            case csl.LoadVarOp(var=var, res=res):
                var = self._var_use(var)
                const = self._var_use(res)
                self.print(f"{const} = {var};")
            case csl.StoreVarOp(var=var, new_value=new_value):
                var = self._var_use(var)
                other = self._var_use(new_value)
                self.print(f"{var} = {other};")
            case csl.PtrCastOp(ptr=ptr, result=result):
                typ = self.mlir_type_to_csl_type(result.type)
                var = self._get_variable_name_for(ptr)
                self._print_or_promote_to_inline_expr(result, f"@ptrcast({typ}, {var})")
            case anyop:
                self.print(f"unknown op {anyop}", prefix="//")

    def print_block(self, body: Block):
        """
        Walks over a block and prints every operation in the block.
        """
        for op in body.ops:
            self.print_op(op)

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
            _cmp_ops=self._cmp_ops,
            _binops=self._binops,
            _counter=self._counter,
            _prefix=self._prefix + self._INDENT,
        )
        self.print("}")


def get_csl_modules_in_module_op(module: ModuleOp) -> Iterable[csl.CslModuleOp]:
    layouts: list[csl.CslModuleOp] = []
    for op in module.body.ops:
        if isinstance(op, csl.CslModuleOp):
            if op.kind.data == csl.ModuleKind.LAYOUT:
                layouts.append(op)
                continue
            yield op
        else:
            warnings.warn("Expected all top-level operations to be `csl.module` ops!")
    yield from layouts


def print_to_csl(
    prog: ModuleOp, output: IO[str], Ctx: type[CslPrintContext] = CslPrintContext
):
    """
    Takes a module op and prints it to the given output stream.
    """
    ctx = Ctx(output)
    ctx.register_binops()

    divider = False
    for module in get_csl_modules_in_module_op(prog):
        if divider:
            ctx.print("// -----")
        divider = True
        ctx.print("// FILE: " + module.sym_name.data)
        ctx.print_block(module.body.block)
