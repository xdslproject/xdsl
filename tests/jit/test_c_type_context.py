import pytest

from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    IndexType,
    IntegerType,
    NoneType,
    Signedness,
)
from xdsl.ir import Attribute
from xdsl.jit.c_type_context import (
    CFuncSignature,
    CTypeContext,
    register_builtin_types,
)
from xdsl.utils.exceptions import JITException


@pytest.fixture
def ctx() -> CTypeContext:
    c = CTypeContext()
    register_builtin_types(c)
    return c


@pytest.mark.parametrize(
    "type_attr, expected",
    [
        (IntegerType(1), "_Bool"),
        (IntegerType(8), "int8_t"),
        (IntegerType(16), "int16_t"),
        (IntegerType(32), "int32_t"),
        (IntegerType(64), "int64_t"),
        (IntegerType(32, Signedness.SIGNED), "int32_t"),
        (IntegerType(32, Signedness.UNSIGNED), "int32_t"),
        (Float32Type(), "float"),
        (Float64Type(), "double"),
        (NoneType(), "void"),
    ],
)
def test_builtin_resolve(ctx: CTypeContext, type_attr: Attribute, expected: str):
    assert ctx.to_c_type(type_attr) == expected


@pytest.mark.parametrize(
    "type_attr, match",
    [
        (IntegerType(0), "integer of width 0"),
        (IntegerType(17), "integer of width 17"),
        (IntegerType(128), "integer of width 128"),
        (Float16Type(), "No C type mapping for type"),
        (IndexType(), "No C type mapping for type"),
    ],
)
def test_builtin_unsupported(ctx: CTypeContext, type_attr: Attribute, match: str):
    with pytest.raises(JITException, match=match):
        ctx.to_c_type(type_attr)


def test_to_c_func_type(ctx: CTypeContext):
    assert ctx.to_c_func_type(
        (Float64Type(), IntegerType(32)), Float64Type()
    ) == CFuncSignature(("double", "int32_t"), "double")


def test_to_c_func_type_propagates_unmapped_type(ctx: CTypeContext):
    with pytest.raises(JITException, match="No C type mapping for type: index"):
        ctx.to_c_func_type((IndexType(),), Float64Type())
