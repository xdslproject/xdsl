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
    AttrSizedOperandSegments,
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


@irdl_attr_definition
class UndefAttr(ParametrizedAttribute):
    """`#cir.undef : T` — typed `undef` constant."""

    name = "cir.undef"

    undef_type: Attribute

    def __init__(self, undef_type: Attribute):
        super().__init__(undef_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.undef_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation(":")
        return [parser.parse_type()]

    def get_type(self) -> Attribute:
        return self.undef_type


@irdl_attr_definition
class PoisonAttr(ParametrizedAttribute):
    """`#cir.poison : T` — typed `poison` constant."""

    name = "cir.poison"

    poison_type: Attribute

    def __init__(self, poison_type: Attribute):
        super().__init__(poison_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.poison_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation(":")
        return [parser.parse_type()]

    def get_type(self) -> Attribute:
        return self.poison_type


@irdl_attr_definition
class ConstRecordAttr(ParametrizedAttribute):
    """`#cir.const_record<{m1, m2, ...}> : !cir.record<...>` — typed
    record/struct initialiser. `members` is an `mlir::ArrayAttr` of typed
    member values, printed inside braces."""

    name = "cir.const_record"

    record_type: Attribute
    members: ArrayAttr[Attribute]

    def __init__(
        self,
        record_type: Attribute,
        members: Sequence[Attribute] | ArrayAttr[Attribute],
    ):
        if not isinstance(members, ArrayAttr):
            members = ArrayAttr(members)
        super().__init__(record_type, members)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.in_braces():
                printer.print_list(self.members.data, printer.print_attribute)
        printer.print_string(" : ")
        printer.print_attribute(self.record_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            members: list[Attribute] = []
            parser.parse_punctuation("{")
            if parser.parse_optional_punctuation("}") is None:
                members.append(parser.parse_attribute())
                while parser.parse_optional_punctuation(",") is not None:
                    members.append(parser.parse_attribute())
                parser.parse_punctuation("}")
        parser.parse_punctuation(":")
        record_type = parser.parse_type()
        return [record_type, ArrayAttr(members)]

    def get_type(self) -> Attribute:
        return self.record_type


@irdl_attr_definition
class GlobalViewAttr(ParametrizedAttribute):
    """`#cir.global_view<@sym (, [i, j, ...])?> : T` — typed pointer to a
    global symbol, optionally with sub-element indices."""

    name = "cir.global_view"

    view_type: Attribute
    symbol: SymbolRefAttr
    indices: ArrayAttr[Attribute]

    def __init__(
        self,
        view_type: Attribute,
        symbol: str | SymbolRefAttr,
        indices: Sequence[Attribute] | ArrayAttr[Attribute] = (),
    ):
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        if not isinstance(indices, ArrayAttr):
            indices = ArrayAttr(indices)
        super().__init__(view_type, symbol, indices)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.symbol)
            if self.indices.data:
                printer.print_string(", [")
                printer.print_list(self.indices.data, printer.print_attribute)
                printer.print_string("]")
        printer.print_string(" : ")
        printer.print_attribute(self.view_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            symbol = parser.parse_attribute()
            indices: list[Attribute] = []
            if parser.parse_optional_punctuation(",") is not None:
                parser.parse_punctuation("[")
                if parser.parse_optional_punctuation("]") is None:
                    indices.append(parser.parse_attribute())
                    while parser.parse_optional_punctuation(",") is not None:
                        indices.append(parser.parse_attribute())
                    parser.parse_punctuation("]")
        parser.parse_punctuation(":")
        view_type = parser.parse_type()
        if not isinstance(symbol, SymbolRefAttr):
            parser.raise_error("expected a symbol reference")
        return [view_type, symbol, ArrayAttr(indices)]

    def get_type(self) -> Attribute:
        return self.view_type


@irdl_attr_definition
class OptInfoAttr(ParametrizedAttribute):
    """`#cir.opt_info<level = N, size = M>` — module-level optimisation flags."""

    name = "cir.opt_info"

    level: IntegerAttr
    size: IntegerAttr

    def __init__(self, level: int, size: int):
        super().__init__(IntegerAttr(level, 32), IntegerAttr(size, 32))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("level = ")
            printer.print_string(str(self.level.value.data))
            printer.print_string(", size = ")
            printer.print_string(str(self.size.value.data))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_keyword("level")
            parser.parse_punctuation("=")
            level = parser.parse_integer()
            parser.parse_punctuation(",")
            parser.parse_keyword("size")
            parser.parse_punctuation("=")
            size = parser.parse_integer()
        return [IntegerAttr(level, 32), IntegerAttr(size, 32)]


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
# Tier-1 ops: extra control flow + lifecycle helpers
# ---------------------------------------------------------------------------


@irdl_op_definition
class BrCondOp(IRDLOperation):
    """`cir.brcond` — conditional branch with two successors."""

    name = "cir.brcond"

    cond = operand_def(BoolType)
    dest_operands_true = var_operand_def()
    dest_operands_false = var_operand_def()
    successors = var_successor_def()
    traits = traits_def(IsTerminator())

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class DoWhileOp(IRDLOperation):
    """`cir.do` — bottom-tested do-while loop. Two regions: body and cond."""

    name = "cir.do"

    body_region = region_def()
    cond_region = region_def()


@irdl_op_definition
class SwitchOp(IRDLOperation):
    """`cir.switch` — structured C/C++ switch."""

    name = "cir.switch"

    condition = operand_def(IntType)
    body = region_def()
    all_enum_cases_covered = opt_prop_def(UnitAttr)


@irdl_op_definition
class CaseOp(IRDLOperation):
    """`cir.case` — case clause within a `cir.switch`."""

    name = "cir.case"

    case_region = region_def()
    value = prop_def(ArrayAttr)
    kind = prop_def(IntegerAttr)


@irdl_op_definition
class SwitchFlatOp(IRDLOperation):
    """`cir.switch.flat` — region-less, LLVM-style switch terminator."""

    name = "cir.switch.flat"

    condition = operand_def(IntType)
    default_operands = var_operand_def()
    case_operands = var_operand_def()
    case_values = prop_def(ArrayAttr)
    case_operand_segments = prop_def(Attribute)
    successors = var_successor_def()
    traits = traits_def(IsTerminator())

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class CopyOp(IRDLOperation):
    """`cir.copy` — typed pointer-to-pointer memcpy."""

    name = "cir.copy"

    dst = operand_def(PointerType)
    src = operand_def(PointerType)
    is_volatile = opt_prop_def(UnitAttr)


@irdl_op_definition
class UnreachableOp(IRDLOperation):
    """`cir.unreachable` — `__builtin_unreachable` / immediate UB."""

    name = "cir.unreachable"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class TrapOp(IRDLOperation):
    """`cir.trap` — `__builtin_trap`."""

    name = "cir.trap"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class ExpectOp(IRDLOperation):
    """`cir.expect` — `__builtin_expect[_with_probability]`."""

    name = "cir.expect"

    val = operand_def(IntType)
    expected = operand_def(IntType)
    res = result_def(IntType)
    prob = opt_prop_def(FloatAttr)


@irdl_op_definition
class AssumeOp(IRDLOperation):
    """`cir.assume` — `__builtin_assume`."""

    name = "cir.assume"

    predicate = operand_def(BoolType)


@irdl_op_definition
class AssumeAlignedOp(IRDLOperation):
    """`cir.assume_aligned` — `__builtin_assume_aligned`."""

    name = "cir.assume_aligned"

    pointer = operand_def(PointerType)
    offset = var_operand_def()
    res = result_def(PointerType)
    alignment = prop_def(IntegerAttr)


@irdl_op_definition
class AssumeSeparateStorageOp(IRDLOperation):
    """`cir.assume_separate_storage` — `__builtin_assume_separate_storage`."""

    name = "cir.assume_separate_storage"

    ptr1 = operand_def(PointerType)
    ptr2 = operand_def(PointerType)


# ---------------------------------------------------------------------------
# Tier-2 ops: builtins, math, vectors, complex, varargs, misc
# ---------------------------------------------------------------------------


@irdl_op_definition
class IsConstantOp(IRDLOperation):
    """`cir.is_constant` — `__builtin_constant_p`."""

    name = "cir.is_constant"

    val = operand_def()
    res = result_def(BoolType)


@irdl_op_definition
class ObjSizeOp(IRDLOperation):
    """`cir.objsize` — `__builtin_object_size`."""

    name = "cir.objsize"

    ptr = operand_def(PointerType)
    res = result_def(IntType)
    min = opt_prop_def(UnitAttr)
    nullunknown = opt_prop_def(UnitAttr)
    dynamic = opt_prop_def(UnitAttr)


@irdl_op_definition
class PtrDiffOp(IRDLOperation):
    """`cir.ptr_diff` — typed pointer subtraction."""

    name = "cir.ptr_diff"

    lhs = operand_def(PointerType)
    rhs = operand_def(PointerType)
    res = result_def(IntType)


@irdl_op_definition
class IsFPClassOp(IRDLOperation):
    """`cir.is_fp_class` — `__builtin_fpclassify` / isnan / isinf."""

    name = "cir.is_fp_class"

    src = operand_def()
    res = result_def(BoolType)
    flags = prop_def(IntegerAttr)


@irdl_op_definition
class PrefetchOp(IRDLOperation):
    """`cir.prefetch` — `__builtin_prefetch`."""

    name = "cir.prefetch"

    addr = operand_def(PointerType)
    locality = opt_prop_def(IntegerAttr)
    is_write = opt_prop_def(UnitAttr, prop_name="isWrite")


@irdl_op_definition
class StackSaveOp(IRDLOperation):
    """`cir.stacksave` — VLA stack snapshot."""

    name = "cir.stacksave"

    res = result_def(PointerType)


@irdl_op_definition
class StackRestoreOp(IRDLOperation):
    """`cir.stackrestore` — restore stack to a previously-saved snapshot."""

    name = "cir.stackrestore"

    ptr = operand_def(PointerType)


@irdl_op_definition
class ReturnAddrOp(IRDLOperation):
    """`cir.return_address` — `__builtin_return_address`."""

    name = "cir.return_address"

    level = operand_def(IntType)
    res = result_def(PointerType)


@irdl_op_definition
class FrameAddrOp(IRDLOperation):
    """`cir.frame_address` — `__builtin_frame_address`."""

    name = "cir.frame_address"

    level = operand_def(IntType)
    res = result_def(PointerType)


@irdl_op_definition
class AddrOfReturnAddrOp(IRDLOperation):
    """`cir.address_of_return_address` — MSVC `_AddressOfReturnAddress`."""

    name = "cir.address_of_return_address"

    res = result_def(PointerType)


@irdl_op_definition
class DynamicCastOp(IRDLOperation):
    """`cir.dyn_cast` — C++ `dynamic_cast`."""

    name = "cir.dyn_cast"

    src = operand_def(PointerType)
    res = result_def(PointerType)
    kind = prop_def(IntegerAttr)
    info = opt_prop_def(Attribute)
    relative_layout = opt_prop_def(UnitAttr)


# Math FP→FP builtins (sqrt, sin, cos, …).


def _make_unary_fp_op(mnemonic: str):
    cls = type(
        f"FP{mnemonic.capitalize()}Op",
        (IRDLOperation,),
        {
            "name": f"cir.{mnemonic}",
            "src": operand_def(),
            "res": result_def(),
            "__doc__": f"`cir.{mnemonic}` — unary FP→FP math builtin.",
        },
    )
    return irdl_op_definition(cls)


SqrtOp = _make_unary_fp_op("sqrt")
ACosOp = _make_unary_fp_op("acos")
ASinOp = _make_unary_fp_op("asin")
ATanOp = _make_unary_fp_op("atan")
CeilOp = _make_unary_fp_op("ceil")
CosOp = _make_unary_fp_op("cos")
ExpOp = _make_unary_fp_op("exp")
Exp2Op = _make_unary_fp_op("exp2")
FAbsOp = _make_unary_fp_op("fabs")
FloorOp = _make_unary_fp_op("floor")
SinOp = _make_unary_fp_op("sin")


# Bit manipulation builtins (`cir.<mnemonic>`).


@irdl_op_definition
class BitClrsbOp(IRDLOperation):
    """`cir.clrsb` — count leading redundant sign bits."""

    name = "cir.clrsb"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitClzOp(IRDLOperation):
    """`cir.clz` — count leading zero bits."""

    name = "cir.clz"
    input = operand_def(IntType)
    result = result_def(IntType)
    poison_zero = opt_prop_def(UnitAttr)


@irdl_op_definition
class BitCtzOp(IRDLOperation):
    """`cir.ctz` — count trailing zero bits."""

    name = "cir.ctz"
    input = operand_def(IntType)
    result = result_def(IntType)
    poison_zero = opt_prop_def(UnitAttr)


@irdl_op_definition
class BitFfsOp(IRDLOperation):
    """`cir.ffs` — find first set bit (1-based)."""

    name = "cir.ffs"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitParityOp(IRDLOperation):
    """`cir.parity` — parity of input bits."""

    name = "cir.parity"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitPopcountOp(IRDLOperation):
    """`cir.popcount` — population count."""

    name = "cir.popcount"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class BitReverseOp(IRDLOperation):
    """`cir.bitreverse` — reverse the bit pattern."""

    name = "cir.bitreverse"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class ByteSwapOp(IRDLOperation):
    """`cir.byte_swap` — byte-order reverse."""

    name = "cir.byte_swap"
    input = operand_def(IntType)
    result = result_def(IntType)


@irdl_op_definition
class RotateOp(IRDLOperation):
    """`cir.rotate` — bit rotation; `rotateLeft` selects direction."""

    name = "cir.rotate"
    input = operand_def(IntType)
    amount = operand_def(IntType)
    result = result_def(IntType)
    rotate_left = opt_prop_def(UnitAttr, prop_name="rotateLeft")


# Complex number ops.


@irdl_op_definition
class ComplexCreateOp(IRDLOperation):
    """`cir.complex.create` — build a complex value from real/imag parts."""

    name = "cir.complex.create"
    real = operand_def()
    imag = operand_def()
    result = result_def(ComplexType)


@irdl_op_definition
class ComplexRealOp(IRDLOperation):
    """`cir.complex.real` — real part of a complex (or scalar pass-through)."""

    name = "cir.complex.real"
    operand = operand_def()
    result = result_def()


@irdl_op_definition
class ComplexImagOp(IRDLOperation):
    """`cir.complex.imag` — imaginary part of a complex."""

    name = "cir.complex.imag"
    operand = operand_def()
    result = result_def()


@irdl_op_definition
class ComplexRealPtrOp(IRDLOperation):
    """`cir.complex.real_ptr` — pointer to the real part of a complex object."""

    name = "cir.complex.real_ptr"
    operand = operand_def(PointerType)
    result = result_def(PointerType)


@irdl_op_definition
class ComplexImagPtrOp(IRDLOperation):
    """`cir.complex.imag_ptr` — pointer to the imaginary part."""

    name = "cir.complex.imag_ptr"
    operand = operand_def(PointerType)
    result = result_def(PointerType)


@irdl_op_definition
class ComplexAddOp(IRDLOperation):
    """`cir.complex.add` — complex addition."""

    name = "cir.complex.add"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)


@irdl_op_definition
class ComplexSubOp(IRDLOperation):
    """`cir.complex.sub` — complex subtraction."""

    name = "cir.complex.sub"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)


@irdl_op_definition
class ComplexMulOp(IRDLOperation):
    """`cir.complex.mul` — complex multiplication; `range` enum encoded as i32."""

    name = "cir.complex.mul"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)
    range = prop_def(IntegerAttr)


@irdl_op_definition
class ComplexDivOp(IRDLOperation):
    """`cir.complex.div` — complex division; `range` enum encoded as i32."""

    name = "cir.complex.div"
    lhs = operand_def(ComplexType)
    rhs = operand_def(ComplexType)
    result = result_def(ComplexType)
    range = prop_def(IntegerAttr)


# Vector ops.


@irdl_op_definition
class VecCreateOp(IRDLOperation):
    """`cir.vec.create` — build a vector value from element operands."""

    name = "cir.vec.create"
    elements = var_operand_def()
    result = result_def(VectorType)


@irdl_op_definition
class VecInsertOp(IRDLOperation):
    """`cir.vec.insert` — replace one element of a vector."""

    name = "cir.vec.insert"
    vec = operand_def(VectorType)
    value = operand_def()
    index = operand_def(IntType)
    result = result_def(VectorType)


@irdl_op_definition
class VecExtractOp(IRDLOperation):
    """`cir.vec.extract` — extract one element from a vector."""

    name = "cir.vec.extract"
    vec = operand_def(VectorType)
    index = operand_def(IntType)
    result = result_def()


@irdl_op_definition
class VecCmpOp(IRDLOperation):
    """`cir.vec.cmp` — element-wise comparison; `kind` encoded as i32."""

    name = "cir.vec.cmp"
    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    result = result_def(VectorType)
    kind = prop_def(IntegerAttr)


@irdl_op_definition
class VecShuffleOp(IRDLOperation):
    """`cir.vec.shuffle` — `__builtin_shufflevector` (compile-time indices)."""

    name = "cir.vec.shuffle"
    vec1 = operand_def(VectorType)
    vec2 = operand_def(VectorType)
    result = result_def(VectorType)
    indices = prop_def(ArrayAttr)


@irdl_op_definition
class VecShuffleDynamicOp(IRDLOperation):
    """`cir.vec.shuffle.dynamic` — `__builtin_shufflevector` (runtime indices)."""

    name = "cir.vec.shuffle.dynamic"
    vec = operand_def(VectorType)
    indices = operand_def(VectorType)
    result = result_def(VectorType)


@irdl_op_definition
class VecTernaryOp(IRDLOperation):
    """`cir.vec.ternary` — element-wise `cond ? a : b` for vectors."""

    name = "cir.vec.ternary"
    cond = operand_def(VectorType)
    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    result = result_def(VectorType)


@irdl_op_definition
class VecSplatOp(IRDLOperation):
    """`cir.vec.splat` — replicate a scalar across a vector."""

    name = "cir.vec.splat"
    value = operand_def()
    result = result_def(VectorType)


# Variadic-arg ops.


@irdl_op_definition
class VAStartOp(IRDLOperation):
    """`cir.va_start` — initialise a `va_list`."""

    name = "cir.va_start"
    arg_list = operand_def(PointerType)
    count = operand_def(IntType)


@irdl_op_definition
class VAEndOp(IRDLOperation):
    """`cir.va_end` — finalise a `va_list`."""

    name = "cir.va_end"
    arg_list = operand_def(PointerType)


@irdl_op_definition
class VACopyOp(IRDLOperation):
    """`cir.va_copy` — copy one `va_list` into another."""

    name = "cir.va_copy"
    src_list = operand_def(PointerType)
    dst_list = operand_def(PointerType)


@irdl_op_definition
class VAArgOp(IRDLOperation):
    """`cir.va_arg` — fetch next variadic argument as a typed value."""

    name = "cir.va_arg"
    arg_list = operand_def(PointerType)
    result = result_def()


# ---------------------------------------------------------------------------
# Dialect
# ---------------------------------------------------------------------------


CIR = Dialect(
    "cir",
    [
        ACosOp,
        ASinOp,
        ATanOp,
        AddrOfReturnAddrOp,
        AllocaOp,
        AssumeAlignedOp,
        AssumeOp,
        AssumeSeparateStorageOp,
        BinOp,
        BitClrsbOp,
        BitClzOp,
        BitCtzOp,
        BitFfsOp,
        BitParityOp,
        BitPopcountOp,
        BitReverseOp,
        BrCondOp,
        BrOp,
        BreakOp,
        ByteSwapOp,
        CallOp,
        CaseOp,
        CastOp,
        CeilOp,
        CmpOp,
        ComplexAddOp,
        ComplexCreateOp,
        ComplexDivOp,
        ComplexImagOp,
        ComplexImagPtrOp,
        ComplexMulOp,
        ComplexRealOp,
        ComplexRealPtrOp,
        ComplexSubOp,
        ConditionOp,
        ConstantOp,
        ContinueOp,
        CopyOp,
        CosOp,
        DoWhileOp,
        DynamicCastOp,
        Exp2Op,
        ExpOp,
        ExpectOp,
        FAbsOp,
        FloorOp,
        ForOp,
        FrameAddrOp,
        FuncOp,
        GetElementOp,
        GetGlobalOp,
        GetMemberOp,
        GlobalOp,
        IfOp,
        IsConstantOp,
        IsFPClassOp,
        LoadOp,
        ObjSizeOp,
        PrefetchOp,
        PtrDiffOp,
        PtrStrideOp,
        ReturnAddrOp,
        ReturnOp,
        RotateOp,
        ScopeOp,
        SelectOp,
        SinOp,
        SqrtOp,
        StackRestoreOp,
        StackSaveOp,
        StoreOp,
        SwitchFlatOp,
        SwitchOp,
        TernaryOp,
        TrapOp,
        UnaryOp,
        UnreachableOp,
        VAArgOp,
        VACopyOp,
        VAEndOp,
        VAStartOp,
        VecCmpOp,
        VecCreateOp,
        VecExtractOp,
        VecInsertOp,
        VecShuffleDynamicOp,
        VecShuffleOp,
        VecSplatOp,
        VecTernaryOp,
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
        ConstRecordAttr,
        DoubleType,
        FP16Type,
        FP80Type,
        FP128Type,
        FuncType,
        GlobalViewAttr,
        IntType,
        LongDoubleType,
        OptInfoAttr,
        PoisonAttr,
        PointerType,
        RecordType,
        SingleType,
        SourceLanguageAttr,
        UndefAttr,
        VectorType,
        VisibilityAttr,
        VoidType,
        ZeroAttr,
    ],
)
