"""
This file contains the definition of the attributes of the WebAssembly dialect.

The paragraphs prefixed by `wasm>` in the documentation of this dialect are
excerpts from the WebAssembly Specification, which is licensed under the terms
described in the WASM-SPEC-LICENSE file.
"""

from collections.abc import Sequence
from enum import auto
from typing import Annotated

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    f32,
    f64,
    i32,
    i64,
)
from xdsl.ir import (
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
)
from xdsl.irdl import (
    Attribute,
    ParameterDef,
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.str_enum import StrEnum

i128 = IntegerType(128)

I32Type = Annotated[IntegerType, i32]
I64Type = Annotated[IntegerType, i64]
I128Type = Annotated[IntegerType, i128]
F32Type = Annotated[IntegerType, f32]
F64Type = Annotated[IntegerType, f64]

I32Attr = IntegerAttr[I32Type]

FuncIdx = I32Attr
TableIdx = I32Attr
MemIdx = I32Attr
GlobalIdx = I32Attr
ElemIdx = I32Attr
DataIdx = I32Attr
LocalIdx = I32Attr
LabelIdx = I32Attr

##==------------------------------------------------------------------------==##
# WebAssembly value types
##==------------------------------------------------------------------------==##


class WasmRefTypeEnum(StrEnum):
    FuncRef = auto()
    """
    wasm> The type funcref denotes the infinite union of all references to
    functions, regardless of their function types.
    """

    ExternRef = auto()
    """
    wasm> The type externref denotes the infinite union of all references to
    objects owned. by the embedder and that can be passed into WebAssembly
    under this type.
    """


class WasmRefType(EnumAttribute[WasmRefTypeEnum], SpacedOpaqueSyntaxAttribute):
    name = "wasm.reftype"


WasmNumType = I32Type | I64Type | F32Type | F64Type
WasmVecType = I128Type
WasmValueType = WasmNumType | WasmVecType | WasmRefType


@irdl_attr_definition
class WasmFuncType(ParametrizedAttribute):
    """
    wasm> Function types classify the signature of functions, mapping a vector
    of parameters to a vector of results. They are also used to classify the
    inputs and outputs of instructions.
    """

    name = "wasm.functype"

    args: ParameterDef[ArrayAttr[WasmValueType]]
    res: ParameterDef[ArrayAttr[WasmValueType]]

    def __init__(self, args: Sequence[WasmValueType], res: Sequence[WasmValueType]):
        super().__init__((ArrayAttr(args), ArrayAttr(res)))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            args = parser.parse_comma_separated_list(
                AttrParser.Delimiter.PAREN, lambda: parser.parse_attribute()
            )
            parser.parse_punctuation("->")
            res = parser.parse_comma_separated_list(
                AttrParser.Delimiter.PAREN, lambda: parser.parse_attribute()
            )
            return (ArrayAttr(args), ArrayAttr(res))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print("(")
            printer.print_list(self.args.data, lambda x: printer.print_attribute(x))
            printer.print(") -> (")
            printer.print_list(self.res.data, lambda x: printer.print_attribute(x))
            printer.print(")")


##==------------------------------------------------------------------------==##
# WebAssembly tables and memories
##==------------------------------------------------------------------------==##


@irdl_attr_definition
class WasmLimits(ParametrizedAttribute):
    """
    wasm> Limits classify the size range of resizable storage associated with
    memory types and table types. If no maximum is given, the respective storage
    can grow to any size.
    """

    name = "wasm.limits"

    min: ParameterDef[I32Attr]
    max: ParameterDef[I32Attr | NoneAttr]


@irdl_attr_definition
class WasmMemoryType(ParametrizedAttribute):
    """
    wasm> Memory types classify linear memories and their size range. The limits
    constrain the minimum and optionally the maximum size of a memory. The limits
    are given in units of page size.
    """

    name = "wasm.mem"

    limits: ParameterDef[WasmLimits]


@irdl_attr_definition
class WasmTableType(ParametrizedAttribute):
    """
    wasm> Table types classify tables over elements of reference type within a
    size range. Like memories, tables are constrained by limits for their
    minimum and optionally maximum size. The limits are given in numbers of
    entries.
    """

    name = "wasm.table"

    elements: ParameterDef[WasmRefType]
    limits: ParameterDef[WasmLimits]


class WasmMutEnum(StrEnum):
    Const = auto()
    Var = auto()


class WasmMut(EnumAttribute[WasmMutEnum], SpacedOpaqueSyntaxAttribute):
    name = "wasm.mut"


@irdl_attr_definition
class WasmGlobalType(ParametrizedAttribute):
    """
    wasm> Global types classify global variables, which hold a value and can
    either be mutable or immutable.
    """

    name = "wasm.globaltype"

    mutability: ParameterDef[WasmMut]
    type: ParameterDef[WasmValueType]


##==------------------------------------------------------------------------==##
# WebAssembly exports
##==------------------------------------------------------------------------==##


@irdl_attr_definition
class WasmExportDescFunc(ParametrizedAttribute):
    name = "wasm.export_desc_func"
    id: ParameterDef[FuncIdx]


@irdl_attr_definition
class WasmExportDescTable(ParametrizedAttribute):
    name = "wasm.export_desc_table"
    id: ParameterDef[TableIdx]


@irdl_attr_definition
class WasmExportDescMem(ParametrizedAttribute):
    name = "wasm.export_desc_mem"
    id: ParameterDef[MemIdx]


@irdl_attr_definition
class WasmExportDescGlobal(ParametrizedAttribute):
    name = "wasm.export_desc_global"
    id: ParameterDef[GlobalIdx]


WasmExportDesc = (
    WasmExportDescFunc | WasmExportDescTable | WasmExportDescMem | WasmExportDescGlobal
)


@irdl_attr_definition
class WasmExport(ParametrizedAttribute):
    """
    wasm> The exports component of a module defines a set of exports that become
    accessible to the host environment once the module has been instantiated.
    Each export is labeled by a unique name. Exportable definitions are
    functions, tables, memories, and globals, which are referenced through a
    respective descriptor.
    """

    name = "wasm.export"

    export_name: ParameterDef[StringAttr]
    desc: ParameterDef[WasmExportDesc]


##==------------------------------------------------------------------------==##
# WebAssembly imports
##==------------------------------------------------------------------------==##


@irdl_attr_definition
class WasmImportDescFunc(ParametrizedAttribute):
    name = "wasm.import_desc_func"
    id: ParameterDef[WasmFuncType]


@irdl_attr_definition
class WasmImportDescTable(ParametrizedAttribute):
    name = "wasm.import_desc_table"
    id: ParameterDef[TableIdx]


@irdl_attr_definition
class WasmImportDescMem(ParametrizedAttribute):
    name = "wasm.import_desc_mem"
    id: ParameterDef[MemIdx]


@irdl_attr_definition
class WasmImportDescGlobal(ParametrizedAttribute):
    name = "wasm.import_desc_global"
    id: ParameterDef[GlobalIdx]


WasmImportDesc = (
    WasmImportDescFunc | WasmImportDescTable | WasmImportDescMem | WasmImportDescGlobal
)


@irdl_attr_definition
class WasmImport(ParametrizedAttribute):
    """
    wasm> The component of a module defines a set of imports that are required
    for instantiation. Each import is labeled by a two-level name space,
    consisting of a module name and a name for an entity within that module.
    Importable definitions are functions, tables, memories, and globals. Each
    import is specified by a descriptor with a respective type that a definition
    provided during instantiation is required to match. Every import defines an
    index in the respective index space. In each index space, the indices of
    imports go before the first index of any definition contained in the module
    itself.

    wasm> Unlike export names, import names are not necessarily unique. It is
    possible to import the same module/name pair multiple times; such imports
    may even have different type descriptions, including different kinds of
    entities. A module with such imports can still be instantiated depending
    on the specifics of how an embedder allows resolving and supplying imports.
    However, embedders are not required to support such overloading, and a
    WebAssembly module itself cannot implement an overloaded name.
    """

    name = "wasm.import"

    import_name: ParameterDef[StringAttr]
    desc: ParameterDef[WasmImportDesc]

    def __init__(self, name: str | StringAttr, desc: WasmImportDesc):
        super().__init__((StringAttr(name) if isinstance(name, str) else name, desc))
