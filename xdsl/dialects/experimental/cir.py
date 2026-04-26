"""
The CIR (ClangIR) dialect.

A direct port of the upstream MLIR CIR dialect from LLVM 22.1.2
(`clang/include/clang/CIR/Dialect/IR/`). This is a fixture-driven port: it
covers the subset of the upstream dialect that the round-trip fixtures in
`c_tests/cir_generic/` actually exercise. See `docs/cir-port/inventory.md` for
the contract and `docs/cir-port/blocked.md` for any fixtures skipped by the
round-trip gate.

The fixtures use MLIR's generic operation form, which means most enum-valued
attributes appear as plain `i32` integer attributes (e.g.
`linkage = 8 : i32`, `kind = 11 : i32`); they are modelled here as
`IntegerAttr` properties rather than typed enum classes.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    Float32Type,
    Float64Type,
    FlatSymbolRefAttr,
    FloatAttr,
    FloatData,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    TypedAttribute,
    UnitAttr,
    i32,
    i64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
    var_successor_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@irdl_attr_definition
class IntType(ParametrizedAttribute, TypeAttribute):
    """`!cir.int<s|u, N>` — CIR arbitrary-precision integer type."""

    name = "cir.int"

    width: IntegerAttr
    is_signed: IntegerAttr  # 0/1 stored as i1 IntegerAttr

    def __init__(self, width: int, is_signed: bool):
        super().__init__(
            IntegerAttr(width, 64),
            IntegerAttr(1 if is_signed else 0, 1),
        )

    @property
    def signed(self) -> bool:
        return bool(self.is_signed.value.data)

    @property
    def bitwidth(self) -> int:
        return self.width.value.data

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("s" if self.signed else "u")
            printer.print_string(", ")
            printer.print_string(str(self.bitwidth))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            sign_kw = parser.parse_identifier()
            if sign_kw not in ("s", "u"):
                parser.raise_error("expected 's' or 'u'")
            parser.parse_punctuation(",")
            width = parser.parse_integer()
            if not (1 <= width <= 128):
                parser.raise_error("integer width must be between 1 and 128")
        return [IntegerAttr(width, 64), IntegerAttr(1 if sign_kw == "s" else 0, 1)]


@irdl_attr_definition
class BoolType(ParametrizedAttribute, TypeAttribute):
    """`!cir.bool` — CIR bool type."""

    name = "cir.bool"


@irdl_attr_definition
class SingleType(ParametrizedAttribute, TypeAttribute):
    """`!cir.float` — CIR single-precision (32-bit) float."""

    name = "cir.float"


@irdl_attr_definition
class DoubleType(ParametrizedAttribute, TypeAttribute):
    """`!cir.double` — CIR double-precision (64-bit) float."""

    name = "cir.double"


@irdl_attr_definition
class FP16Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.f16` — CIR half-precision (16-bit) float."""

    name = "cir.f16"


@irdl_attr_definition
class BF16Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.bf16` — CIR bfloat16 (16-bit) float."""

    name = "cir.bf16"


@irdl_attr_definition
class FP80Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.f80` — CIR x87 80-bit extended-precision float."""

    name = "cir.f80"


@irdl_attr_definition
class FP128Type(ParametrizedAttribute, TypeAttribute):
    """`!cir.f128` — CIR quad-precision (128-bit) float."""

    name = "cir.f128"


@irdl_attr_definition
class LongDoubleType(ParametrizedAttribute, TypeAttribute):
    """`!cir.long_double<T>` — CIR `long double`, parametric on its underlying
    floating-point format (one of `!cir.double`, `!cir.f80`, `!cir.f128`)."""

    name = "cir.long_double"

    underlying: Attribute

    def __init__(self, underlying: Attribute):
        super().__init__(underlying)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.underlying)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            underlying = parser.parse_type()
            if not isinstance(underlying, (DoubleType, FP80Type, FP128Type)):
                parser.raise_error(
                    "expected !cir.double, !cir.f80 or !cir.f128 underlying type"
                )
        return [underlying]


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, TypeAttribute):
    """`!cir.complex<T>` — CIR `_Complex` type. The element type must be a CIR
    integer or floating-point type."""

    name = "cir.complex"

    element_type: Attribute

    def __init__(self, element_type: Attribute):
        super().__init__(element_type)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.element_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            element_type = parser.parse_type()
        return [element_type]


@irdl_attr_definition
class VectorType(ParametrizedAttribute, TypeAttribute):
    """`!cir.vector<N x T>` (fixed) or `!cir.vector<[N] x T>` (scalable) — CIR
    one-dimensional vector type."""

    name = "cir.vector"

    element_type: Attribute
    size: IntegerAttr
    is_scalable: IntegerAttr

    def __init__(
        self, element_type: Attribute, size: int, is_scalable: bool = False
    ):
        super().__init__(
            element_type,
            IntegerAttr(size, 64),
            IntegerAttr(1 if is_scalable else 0, 1),
        )

    @property
    def length(self) -> int:
        return self.size.value.data

    @property
    def scalable(self) -> bool:
        return bool(self.is_scalable.value.data)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if self.scalable:
                printer.print_string("[")
            printer.print_string(str(self.length))
            if self.scalable:
                printer.print_string("]")
            printer.print_string(" x ")
            printer.print_attribute(self.element_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            is_scalable = parser.parse_optional_punctuation("[") is not None
            size = parser.parse_integer()
            if is_scalable:
                parser.parse_punctuation("]")
            parser.parse_keyword("x")
            element_type = parser.parse_type()
        return [
            element_type,
            IntegerAttr(size, 64),
            IntegerAttr(1 if is_scalable else 0, 1),
        ]


@irdl_attr_definition
class VoidType(ParametrizedAttribute, TypeAttribute):
    """`!cir.void` — CIR void type."""

    name = "cir.void"


@irdl_attr_definition
class PointerType(ParametrizedAttribute, TypeAttribute):
    """`!cir.ptr<T>` — CIR pointer type. Address space is not modelled."""

    name = "cir.ptr"

    pointee: Attribute

    def __init__(self, pointee: Attribute):
        super().__init__(pointee)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.pointee)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            pointee = parser.parse_type()
            # addrspace clause: `, target_address_space(N)` — not present in fixtures
        return [pointee]


@irdl_attr_definition
class ArrayType(ParametrizedAttribute, TypeAttribute):
    """`!cir.array<T x N>` — CIR fixed-size array type."""

    name = "cir.array"

    element_type: Attribute
    size: IntegerAttr

    def __init__(self, element_type: Attribute, size: int):
        super().__init__(element_type, IntegerAttr(size, 64))

    @property
    def length(self) -> int:
        return self.size.value.data

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.element_type)
            printer.print_string(" x ")
            printer.print_string(str(self.length))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            element_type = parser.parse_type()
            parser.parse_shape_delimiter()
            size = parser.parse_integer()
        return [element_type, IntegerAttr(size, 64)]


@irdl_attr_definition
class FuncType(ParametrizedAttribute, TypeAttribute):
    """`!cir.func<(T1, T2, ...) -> R>` — CIR function type.

    The return type is optional; absence means a `void`-like return. A trailing
    `...` in the parameter list marks the function as variadic.
    """

    name = "cir.func"

    inputs: ArrayAttr[Attribute]
    return_type: Attribute  # VoidType used as sentinel for "no return"
    is_var_arg: IntegerAttr

    def __init__(
        self,
        inputs: Sequence[Attribute] | ArrayAttr[Attribute],
        return_type: Attribute | None = None,
        is_var_arg: bool = False,
    ):
        if not isinstance(inputs, ArrayAttr):
            inputs = ArrayAttr(inputs)
        if return_type is None:
            return_type = VoidType()
        super().__init__(
            inputs, return_type, IntegerAttr(1 if is_var_arg else 0, 1)
        )

    @property
    def has_void_return(self) -> bool:
        return isinstance(self.return_type, VoidType)

    @property
    def varargs(self) -> bool:
        return bool(self.is_var_arg.value.data)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.in_parens():
                printer.print_list(self.inputs.data, printer.print_attribute)
                if self.varargs:
                    if self.inputs.data:
                        printer.print_string(", ")
                    printer.print_string("...")
            if not self.has_void_return:
                printer.print_string(" -> ")
                printer.print_attribute(self.return_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            inputs: list[Attribute] = []
            is_var_arg = False
            parser.parse_punctuation("(")
            if parser.parse_optional_punctuation(")") is None:
                while True:
                    if parser.parse_optional_punctuation("...") is not None:
                        is_var_arg = True
                        parser.parse_punctuation(")")
                        break
                    inputs.append(parser.parse_type())
                    if parser.parse_optional_punctuation(",") is None:
                        parser.parse_punctuation(")")
                        break
            return_type: Attribute = VoidType()
            if parser.parse_optional_punctuation("->") is not None:
                return_type = parser.parse_type()
        return [
            ArrayAttr(inputs),
            return_type,
            IntegerAttr(1 if is_var_arg else 0, 1),
        ]


@irdl_attr_definition
class RecordType(ParametrizedAttribute, TypeAttribute):
    """`!cir.record<struct|union|class "name"? (packed)? (padded)? (incomplete | { members })>`.

    The fixtures only exercise complete identified records, so mutability /
    self-reference is not modelled here.
    """

    name = "cir.record"

    members: ArrayAttr[Attribute]
    record_name: StringAttr  # empty string means anonymous
    kind: StringAttr  # "struct" | "union" | "class"
    is_incomplete: IntegerAttr
    is_packed: IntegerAttr
    is_padded: IntegerAttr

    def __init__(
        self,
        members: Sequence[Attribute] | ArrayAttr[Attribute],
        record_name: str | StringAttr = "",
        kind: str | StringAttr = "struct",
        incomplete: bool = False,
        packed: bool = False,
        padded: bool = False,
    ):
        if not isinstance(members, ArrayAttr):
            members = ArrayAttr(members)
        if isinstance(record_name, str):
            record_name = StringAttr(record_name)
        if isinstance(kind, str):
            kind = StringAttr(kind)
        super().__init__(
            members,
            record_name,
            kind,
            IntegerAttr(1 if incomplete else 0, 1),
            IntegerAttr(1 if packed else 0, 1),
            IntegerAttr(1 if padded else 0, 1),
        )

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.kind.data)
            if self.record_name.data:
                printer.print_string(" ")
                printer.print_string_literal(self.record_name.data)
            printer.print_string(" ")
            if self.is_packed.value.data:
                printer.print_string("packed ")
            if self.is_padded.value.data:
                printer.print_string("padded ")
            if self.is_incomplete.value.data:
                printer.print_string("incomplete")
            else:
                with printer.in_braces():
                    printer.print_list(self.members.data, printer.print_attribute)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            kind = parser.parse_identifier()
            if kind not in ("struct", "union", "class"):
                parser.raise_error("expected 'struct', 'union' or 'class'")
            record_name = parser.parse_optional_str_literal() or ""
            packed = parser.parse_optional_keyword("packed") is not None
            padded = parser.parse_optional_keyword("padded") is not None
            members: list[Attribute] = []
            incomplete = parser.parse_optional_keyword("incomplete") is not None
            if not incomplete:
                parser.parse_punctuation("{")
                if parser.parse_optional_punctuation("}") is None:
                    members.append(parser.parse_type())
                    while parser.parse_optional_punctuation(",") is not None:
                        members.append(parser.parse_type())
                    parser.parse_punctuation("}")
        return [
            ArrayAttr(members),
            StringAttr(record_name),
            StringAttr(kind),
            IntegerAttr(1 if incomplete else 0, 1),
            IntegerAttr(1 if packed else 0, 1),
            IntegerAttr(1 if padded else 0, 1),
        ]


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


@irdl_attr_definition
class CIRBoolAttr(ParametrizedAttribute):
    """`#cir.bool<true|false>`."""

    name = "cir.bool"

    bool_type: BoolType
    value: IntegerAttr

    def __init__(self, value: bool):
        super().__init__(BoolType(), IntegerAttr(1 if value else 0, 1))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("true" if self.value.value.data else "false")
        printer.print_string(" : ")
        printer.print_attribute(self.bool_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            kw = parser.parse_identifier()
            if kw not in ("true", "false"):
                parser.raise_error("expected 'true' or 'false'")
        parser.parse_punctuation(":")
        bool_type = parser.parse_type()
        if not isinstance(bool_type, BoolType):
            parser.raise_error("expected !cir.bool type")
        return [bool_type, IntegerAttr(1 if kw == "true" else 0, 1)]

    def get_type(self) -> Attribute:
        return self.bool_type


@irdl_attr_definition
class CIRIntAttr(ParametrizedAttribute):
    """`#cir.int<N> : !cir.int<...>` — typed integer constant."""

    name = "cir.int"

    int_type: IntType
    value: IntegerAttr

    def __init__(self, value: int, int_type: IntType):
        super().__init__(int_type, IntegerAttr(value, 64))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            v = self.value.value.data
            if self.int_type.signed:
                width = self.int_type.bitwidth
                if v >= (1 << (width - 1)):
                    v -= 1 << width
            printer.print_string(str(v))
        printer.print_string(" : ")
        printer.print_attribute(self.int_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            value = parser.parse_integer(allow_negative=True)
        parser.parse_punctuation(":")
        int_type = parser.parse_type()
        if not isinstance(int_type, IntType):
            parser.raise_error("expected !cir.int type for #cir.int attribute")
        return [int_type, IntegerAttr(value, 64)]

    def get_type(self) -> Attribute:
        return self.int_type


@irdl_attr_definition
class CIRFPAttr(ParametrizedAttribute):
    """`#cir.fp<F> : !cir.<float|double>` — typed float constant."""

    name = "cir.fp"

    fp_type: Attribute  # any CIR FP type
    value: FloatAttr

    def __init__(self, value: float, fp_type: Attribute):
        builtin_fp = _cir_fp_to_builtin(fp_type)
        super().__init__(fp_type, FloatAttr(value, builtin_fp))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_float(self.value.value.data, self.value.type)
        printer.print_string(" : ")
        printer.print_attribute(self.fp_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            f = parser.parse_float()
        parser.parse_punctuation(":")
        fp_type = parser.parse_type()
        if not isinstance(
            fp_type,
            (
                SingleType,
                DoubleType,
                FP16Type,
                BF16Type,
                FP80Type,
                FP128Type,
                LongDoubleType,
            ),
        ):
            parser.raise_error("expected a CIR floating-point type for #cir.fp")
        builtin_fp = _cir_fp_to_builtin(fp_type)
        return [fp_type, FloatAttr(f, builtin_fp)]

    def get_type(self) -> Attribute:
        return self.fp_type


def _cir_fp_to_builtin(t: Attribute) -> AnyFloat:
    from xdsl.dialects.builtin import (
        BFloat16Type,
        Float16Type,
        Float80Type,
        Float128Type,
    )

    if isinstance(t, SingleType):
        return Float32Type()
    if isinstance(t, DoubleType):
        return Float64Type()
    if isinstance(t, FP16Type):
        return Float16Type()
    if isinstance(t, BF16Type):
        return BFloat16Type()
    if isinstance(t, FP80Type):
        return Float80Type()
    if isinstance(t, FP128Type):
        return Float128Type()
    if isinstance(t, LongDoubleType):
        return _cir_fp_to_builtin(t.underlying)
    raise ValueError(f"unsupported CIR FP type {t}")


@irdl_attr_definition
class ZeroAttr(ParametrizedAttribute):
    """`#cir.zero : T` — zero-initialiser."""

    name = "cir.zero"

    zero_type: Attribute

    def __init__(self, zero_type: Attribute):
        super().__init__(zero_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.zero_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation(":")
        zero_type = parser.parse_type()
        return [zero_type]

    def get_type(self) -> Attribute:
        return self.zero_type


@irdl_attr_definition
class ConstPtrAttr(ParametrizedAttribute):
    """`#cir.ptr<null>` or `#cir.ptr<N>` — typed pointer constant."""

    name = "cir.ptr"

    ptr_type: PointerType
    value: IntegerAttr

    def __init__(self, value: int, ptr_type: PointerType):
        super().__init__(ptr_type, IntegerAttr(value, 64))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if self.value.value.data == 0:
                printer.print_string("null")
            else:
                printer.print_int(self.value.value.data, IntegerType(64))
        printer.print_string(" : ")
        printer.print_attribute(self.ptr_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            if parser.parse_optional_keyword("null") is not None:
                value = 0
            else:
                value = parser.parse_integer(allow_negative=True)
        parser.parse_punctuation(":")
        ptr_type = parser.parse_type()
        if not isinstance(ptr_type, PointerType):
            parser.raise_error("expected !cir.ptr type for #cir.ptr attribute")
        return [ptr_type, IntegerAttr(value, 64)]

    def get_type(self) -> Attribute:
        return self.ptr_type


@irdl_attr_definition
class ConstArrayAttr(ParametrizedAttribute):
    """`#cir.const_array<elts : !cir.array<T x N>>` — array constant.

    `elts` is either a string literal (for char arrays) or an `[...]` array of
    typed attributes.
    """

    name = "cir.const_array"

    arr_type: Attribute
    elts: Attribute  # StringAttr or ArrayAttr
    trailing_zeros: IntegerAttr

    def __init__(
        self,
        arr_type: Attribute,
        elts: Attribute,
        trailing_zeros: int = 0,
    ):
        super().__init__(arr_type, elts, IntegerAttr(trailing_zeros, 64))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if isinstance(self.elts, StringAttr):
                # MLIR-style hex escapes for string payloads.
                printer.print_bytes_literal(
                    self.elts.data.encode("utf-8", "surrogateescape")
                )
                # String literals carry their type via the inner `: T`.
                printer.print_string(" : ")
                printer.print_attribute(self.arr_type)
            else:
                printer.print_attribute(self.elts)
        # Outer typed-attribute `: T` suffix (TypedAttrInterface).
        printer.print_string(" : ")
        printer.print_attribute(self.arr_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            elts = parser.parse_attribute()
            inner_type = None
            # Inner `: T` only present when elts is a string literal.
            if parser.parse_optional_punctuation(":") is not None:
                inner_type = parser.parse_type()
        # Outer `: T` typed-attribute suffix.
        parser.parse_punctuation(":")
        outer_type = parser.parse_type()
        arr_type = inner_type if inner_type is not None else outer_type
        zeros = 0
        if isinstance(arr_type, ArrayType):
            if isinstance(elts, StringAttr):
                zeros = max(0, arr_type.length - len(elts.data))
            elif isinstance(elts, ArrayAttr):
                zeros = max(0, arr_type.length - len(elts.data))
        return [arr_type, elts, IntegerAttr(zeros, 64)]

    def get_type(self) -> Attribute:
        return self.arr_type


@irdl_attr_definition
class SourceLanguageAttr(ParametrizedAttribute):
    """`#cir.lang<c>` or `#cir<lang c>` — source language enum."""

    name = "cir.lang"

    value: StringAttr

    def __init__(self, value: str | StringAttr):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__(value)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.value.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        # Both opaque (`#cir<lang c>`) and pretty (`#cir.lang<c>`) forms reach
        # this code with the cursor either on `<` (pretty) or just past the
        # bare-ident `lang` (opaque).
        if parser.parse_optional_punctuation("<") is not None:
            kw = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            kw = parser.parse_identifier()
        return [StringAttr(kw)]


@irdl_attr_definition
class VisibilityAttr(ParametrizedAttribute):
    """`#cir.visibility<default>` or `#cir<visibility default>` — visibility kind."""

    name = "cir.visibility"

    value: StringAttr

    def __init__(self, value: str | StringAttr):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__(value)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.value.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        if parser.parse_optional_punctuation("<") is not None:
            kw = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            kw = parser.parse_identifier()
        return [StringAttr(kw)]


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@irdl_op_definition
class AllocaOp(IRDLOperation):
    """`cir.alloca` — scope-local stack allocation."""

    name = "cir.alloca"

    dyn_alloc_size = var_operand_def()
    addr = result_def(PointerType)

    alloca_type = prop_def(Attribute, prop_name="allocaType")
    alloc_name = opt_prop_def(StringAttr, prop_name="name")
    init = opt_prop_def(UnitAttr)
    constant = opt_prop_def(UnitAttr)
    alignment = opt_prop_def(IntegerAttr)
    annotations = opt_prop_def(ArrayAttr, prop_name="annotations")


@irdl_op_definition
class GlobalOp(IRDLOperation):
    """`cir.global` — module-level global variable or function declaration body."""

    name = "cir.global"

    sym_name = prop_def(StringAttr)
    sym_type = prop_def(Attribute)
    sym_visibility = opt_prop_def(StringAttr)
    linkage = prop_def(IntegerAttr)
    global_visibility = prop_def(VisibilityAttr)
    alignment = opt_prop_def(IntegerAttr)
    constant = opt_prop_def(UnitAttr)
    dso_local = opt_prop_def(UnitAttr)
    initial_value = opt_prop_def(Attribute)
    tls_model = opt_prop_def(IntegerAttr, prop_name="tls_model")
    section = opt_prop_def(StringAttr)
    comdat = opt_prop_def(UnitAttr)
    annotations = opt_prop_def(ArrayAttr)
    addr_space = opt_prop_def(Attribute, prop_name="addr_space")
    ctor = opt_prop_def(Attribute)
    dtor = opt_prop_def(Attribute)

    body = region_def()
    ctor_region = region_def()


@irdl_op_definition
class FuncOp(IRDLOperation):
    """`cir.func` — function definition or declaration."""

    name = "cir.func"

    sym_name = prop_def(StringAttr)
    function_type = prop_def(FuncType)
    sym_visibility = opt_prop_def(StringAttr)
    linkage = prop_def(IntegerAttr)
    global_visibility = prop_def(VisibilityAttr)
    inline_kind = opt_prop_def(IntegerAttr)
    dso_local = opt_prop_def(UnitAttr)
    builtin = opt_prop_def(UnitAttr)
    coroutine = opt_prop_def(UnitAttr)
    lambda_ = opt_prop_def(UnitAttr, prop_name="lambda")
    no_proto = opt_prop_def(UnitAttr)
    extra_attrs = opt_prop_def(Attribute)
    arg_attrs = opt_prop_def(ArrayAttr)
    res_attrs = opt_prop_def(ArrayAttr)
    aliasee = opt_prop_def(Attribute)
    global_ctor_priority = opt_prop_def(IntegerAttr)
    global_dtor_priority = opt_prop_def(IntegerAttr)
    cxx_special_member = opt_prop_def(Attribute)
    annotations = opt_prop_def(ArrayAttr)
    comdat = opt_prop_def(UnitAttr)

    body = region_def()


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """`cir.return` — return from a function."""

    name = "cir.return"

    arguments = var_operand_def()
    traits = traits_def(IsTerminator())


@irdl_op_definition
class CallOp(IRDLOperation):
    """`cir.call` — call a function."""

    name = "cir.call"

    arg_ops = var_operand_def()
    results_ = var_result_def()

    callee = opt_prop_def(FlatSymbolRefAttr)
    side_effect = opt_prop_def(IntegerAttr)
    nothrow = opt_prop_def(UnitAttr)
    exception = opt_prop_def(UnitAttr)
    extra_attrs = opt_prop_def(Attribute)
    calling_conv = opt_prop_def(Attribute)
    arg_attrs = opt_prop_def(ArrayAttr)
    res_attrs = opt_prop_def(ArrayAttr)


@irdl_op_definition
class GetGlobalOp(IRDLOperation):
    """`cir.get_global` — take the address of a global symbol."""

    name = "cir.get_global"

    addr = result_def(PointerType)
    glob_name = prop_def(FlatSymbolRefAttr, prop_name="name")
    tls = opt_prop_def(UnitAttr)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """`cir.const` — materialise a constant attribute as an SSA value."""

    name = "cir.const"

    res = result_def()
    value = prop_def(Attribute)


@irdl_op_definition
class LoadOp(IRDLOperation):
    """`cir.load` — load through a pointer."""

    name = "cir.load"

    addr = operand_def(PointerType)
    res = result_def()

    alignment = opt_prop_def(IntegerAttr)
    is_volatile = opt_prop_def(UnitAttr)
    is_nontemporal = opt_prop_def(UnitAttr)
    is_deref = opt_prop_def(UnitAttr, prop_name="isDeref")
    mem_order = opt_prop_def(IntegerAttr)
    sync_scope = opt_prop_def(IntegerAttr)
    tbaa = opt_prop_def(Attribute)


@irdl_op_definition
class StoreOp(IRDLOperation):
    """`cir.store` — store a value through a pointer."""

    name = "cir.store"

    value = operand_def()
    addr = operand_def(PointerType)

    alignment = opt_prop_def(IntegerAttr)
    is_volatile = opt_prop_def(UnitAttr)
    is_nontemporal = opt_prop_def(UnitAttr)
    mem_order = opt_prop_def(IntegerAttr)
    sync_scope = opt_prop_def(IntegerAttr)
    tbaa = opt_prop_def(Attribute)


@irdl_op_definition
class CastOp(IRDLOperation):
    """`cir.cast` — type conversion. The cast kind is encoded as `kind : i32`."""

    name = "cir.cast"

    src = operand_def()
    res = result_def()
    kind = prop_def(IntegerAttr)


@irdl_op_definition
class UnaryOp(IRDLOperation):
    """`cir.unary` — unary arithmetic. Kind encoded as `kind : i32`."""

    name = "cir.unary"

    input = operand_def()
    res = result_def()

    kind = prop_def(IntegerAttr)
    no_signed_wrap = opt_prop_def(UnitAttr)
    no_unsigned_wrap = opt_prop_def(UnitAttr)


@irdl_op_definition
class BinOp(IRDLOperation):
    """`cir.binop` — binary arithmetic. Kind encoded as `kind : i32`."""

    name = "cir.binop"

    lhs = operand_def()
    rhs = operand_def()
    res = result_def()

    kind = prop_def(IntegerAttr)
    no_signed_wrap = opt_prop_def(UnitAttr)
    no_unsigned_wrap = opt_prop_def(UnitAttr)


@irdl_op_definition
class CmpOp(IRDLOperation):
    """`cir.cmp` — comparison. Predicate encoded as `kind : i32`."""

    name = "cir.cmp"

    lhs = operand_def()
    rhs = operand_def()
    res = result_def(BoolType)
    kind = prop_def(IntegerAttr)


@irdl_op_definition
class IfOp(IRDLOperation):
    """`cir.if` — conditional branch with two regions."""

    name = "cir.if"

    cond = operand_def(BoolType)

    then_region = region_def()
    else_region = region_def()


@irdl_op_definition
class ScopeOp(IRDLOperation):
    """`cir.scope` — lexical scope with optional yielded results."""

    name = "cir.scope"

    results_ = var_result_def()
    body = region_def()


@irdl_op_definition
class WhileOp(IRDLOperation):
    """`cir.while` — top-tested loop. Two regions: condition and body."""

    name = "cir.while"

    cond_region = region_def()
    body_region = region_def()


@irdl_op_definition
class ForOp(IRDLOperation):
    """`cir.for` — three-region C-style for-loop (cond, body, step)."""

    name = "cir.for"

    cond_region = region_def()
    body_region = region_def()
    step_region = region_def()


@irdl_op_definition
class ConditionOp(IRDLOperation):
    """`cir.condition` — terminator for the cond-region of `cir.while`/`cir.for`."""

    name = "cir.condition"

    cond = operand_def(BoolType)
    traits = traits_def(IsTerminator())


@irdl_op_definition
class YieldOp(IRDLOperation):
    """`cir.yield` — terminator for structured control-flow regions."""

    name = "cir.yield"

    arguments = var_operand_def()
    traits = traits_def(IsTerminator())


@irdl_op_definition
class BreakOp(IRDLOperation):
    """`cir.break` — terminator transferring control out of a loop body."""

    name = "cir.break"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class ContinueOp(IRDLOperation):
    """`cir.continue` — terminator transferring control to the loop step."""

    name = "cir.continue"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class BrOp(IRDLOperation):
    """`cir.br` — unconditional branch with optional block-arg payload."""

    name = "cir.br"

    arguments = var_operand_def()
    successor = var_successor_def()
    traits = traits_def(IsTerminator())


@irdl_op_definition
class PtrStrideOp(IRDLOperation):
    """`cir.ptr_stride` — pointer + element-stride offset."""

    name = "cir.ptr_stride"

    base = operand_def(PointerType)
    stride = operand_def()
    res = result_def(PointerType)


@irdl_op_definition
class GetElementOp(IRDLOperation):
    """`cir.get_element` — array element address."""

    name = "cir.get_element"

    base = operand_def(PointerType)
    index = operand_def()
    res = result_def(PointerType)


@irdl_op_definition
class GetMemberOp(IRDLOperation):
    """`cir.get_member` — record field address."""

    name = "cir.get_member"

    addr = operand_def(PointerType)
    res = result_def(PointerType)

    index_attr = prop_def(IntegerAttr)
    member_name = prop_def(StringAttr, prop_name="name")


@irdl_op_definition
class TernaryOp(IRDLOperation):
    """`cir.ternary` — `?:`-style structured choice with two regions."""

    name = "cir.ternary"

    cond = operand_def(BoolType)
    results_ = var_result_def()

    true_region = region_def()
    false_region = region_def()


@irdl_op_definition
class SelectOp(IRDLOperation):
    """`cir.select` — value-level select."""

    name = "cir.select"

    cond = operand_def(BoolType)
    true_value = operand_def()
    false_value = operand_def()
    res = result_def()


# ---------------------------------------------------------------------------
# Dialect
# ---------------------------------------------------------------------------


CIR = Dialect(
    "cir",
    [
        AllocaOp,
        BinOp,
        BrOp,
        BreakOp,
        CallOp,
        CastOp,
        CmpOp,
        ConditionOp,
        ConstantOp,
        ContinueOp,
        ForOp,
        FuncOp,
        GetElementOp,
        GetGlobalOp,
        GetMemberOp,
        GlobalOp,
        IfOp,
        LoadOp,
        PtrStrideOp,
        ReturnOp,
        ScopeOp,
        SelectOp,
        StoreOp,
        TernaryOp,
        UnaryOp,
        WhileOp,
        YieldOp,
    ],
    [
        ArrayType,
        BF16Type,
        BoolType,
        CIRBoolAttr,
        CIRFPAttr,
        CIRIntAttr,
        ComplexType,
        ConstArrayAttr,
        ConstPtrAttr,
        DoubleType,
        FP16Type,
        FP80Type,
        FP128Type,
        FuncType,
        IntType,
        LongDoubleType,
        PointerType,
        RecordType,
        SingleType,
        SourceLanguageAttr,
        VectorType,
        VisibilityAttr,
        VoidType,
        ZeroAttr,
    ],
)
