"""
This file contains the definition of the WebAssembly (wasm) dialect.

The purpose of this dialect is to model WebAssembly modules at the lowest
possible level, as per the WebAssembly Specification 2.0.

In particular, importing and exporting a WebAssembly binary module through this
dialect should yield bit-wise identical results. TODO: say it's the encoding

The paragraphs prefixed by `wasm>` in the documentation of this dialect are
excerpts from the WebAssembly Specification, which is licensed under the terms
at the bottom of this file.
"""

from enum import auto
from typing import Annotated

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    f32,
    f64,
    i32,
    i64,
)
from xdsl.ir import (
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
)
from xdsl.utils.str_enum import StrEnum

i128 = IntegerType(128)

I32Type = Annotated[IntegerType, i32]
I64Type = Annotated[IntegerType, i64]
I128Type = Annotated[IntegerType, i128]
F32Type = Annotated[IntegerType, f32]
F64Type = Annotated[IntegerType, f64]

I32Attr = IntegerAttr[I32Type]

TypeIdx = I32Attr
FuncIdx = I32Attr
TableIdx = I32Attr
MemIdx = I32Attr
GlobalIdx = I32Attr
ElemIdx = I32Attr
DataIdx = I32Attr
LocalIdx = I32Attr
LabelIdx = I32Attr


# TODO: group operations via big comment separators


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


@irdl_attr_definition
class WasmLimits(ParametrizedAttribute):
    """
    wasm> Limits classify the size range of resizable storage associated with
    memory types and table types. If no maximum is given, the respective storage
    can grow to any size.
    """

    name = "wasm.limits"

    min: ParameterDef[I32Attr]
    max: ParameterDef[I32Attr] | None


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


@irdl_attr_definition
class WasmImportDescFunc(ParametrizedAttribute):
    name = "wasm.import_desc_func"
    id: ParameterDef[FuncIdx]


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


@irdl_op_definition
class WasmTypeSec(IRDLOperation):
    name = "wasm.type_sec"

    func_types: ArrayAttr[WasmFuncType] = prop_def(ArrayAttr[WasmFuncType])


@irdl_op_definition
class WasmImportSec(IRDLOperation):
    name = "wasm.import_sec"

    imports: ArrayAttr[WasmImport] = prop_def(ArrayAttr[WasmImport])


@irdl_op_definition
class WasmFuncSec(IRDLOperation):
    name = "wasm.func_sec"

    types: ArrayAttr[TypeIdx] = prop_def(ArrayAttr[TypeIdx])


@irdl_op_definition
class WasmTableSec(IRDLOperation):
    name = "wasm.table_sec"

    tables: ArrayAttr[WasmTableType] = prop_def(ArrayAttr[WasmTableType])


@irdl_op_definition
class WasmMemSec(IRDLOperation):
    name = "wasm.mem_sec"

    tables: ArrayAttr[WasmMemoryType] = prop_def(ArrayAttr[WasmMemoryType])


@irdl_op_definition
class WasmGlobalSec(IRDLOperation):
    name = "wasm.global_sec"

    body: Region = region_def("single_block")  # TODO


@irdl_op_definition
class WasmExportSec(IRDLOperation):
    name = "wasm.export_sec"

    exports: ArrayAttr[WasmExport] = prop_def(ArrayAttr[WasmExport])


@irdl_op_definition
class WasmStartSec(IRDLOperation):
    name = "wasm.start_sec"

    start: FuncIdx | None = opt_prop_def(FuncIdx)


@irdl_op_definition
class WasmElemSec(IRDLOperation):
    name = "wasm.elem_sec"

    body: Region = region_def("single_block")  # TODO


@irdl_op_definition
class WasmDataCountSec(IRDLOperation):
    name = "wasm.datacount_sec"

    count: I32Attr | None = opt_prop_def(I32Attr)


@irdl_op_definition
class WasmCodeSec(IRDLOperation):
    name = "wasm.code_sec"

    body: Region = region_def("single_block")  # TODO


@irdl_op_definition
class WasmDataSec(IRDLOperation):
    name = "wasm.data_sec"

    body: Region = region_def("single_block")  # TODO


@irdl_op_definition
class WasmModule(IRDLOperation):
    """
    wasm> WebAssembly programs are organized into modules, which are the unit of
    deployment, loading, and compilation. A module collects definitions for
    types, functions, tables, memories, and globals. In addition, it can
    declare imports and exports and provide initialization in the form of
    data and element segments, or a start function.

    The `body` region stores the different sections of the wasm binary file.
    This is achieved via regions instead of attributes in order to handle
    the ordering of custom sections while still being able to model WebAssembly
    expressions as operations.

    The wasm binary format enforces the ordering of non-custom sections, which
    is also verified by this operation.
    """

    name = "wasm.module"

    # TODO: verifier

    """
    types: ArrayAttr[WasmFuncType] = prop_def(ArrayAttr[WasmFuncType])
    tables: ArrayAttr[WasmTableType] = prop_def(ArrayAttr[WasmTableType])
    mems: ArrayAttr[WasmLimits] = prop_def(ArrayAttr[WasmLimits])
    start: FuncIdx | None = opt_prop_def(FuncIdx)
    exports: ArrayAttr[WasmExport] = prop_def(ArrayAttr[WasmExport])
    imports: ArrayAttr[WasmImport] = prop_def(ArrayAttr[WasmImport])
    """

    body: Region = region_def("single_block")


Wasm = Dialect("wasm", [], [])  # TODO
"""
The WebAssembly dialect.
"""

"""
--- Licensing Terms for the WebAssembly Specification
--- Only applies to the parts of the documentation specified in the header
--- of this file.

WebAssembly Specification: https://webassembly.github.io/spec/core/index.html

    Copyright © 2024 World Wide Web Consortium.
    All Rights Reserved. This work is distributed under the
    W3C® Software and Document License [1] in the hope that it
    will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.

    [1] https://www.w3.org/Consortium/Legal/copyright-software


This work is being provided by the copyright holders under the following
license.

License

By obtaining and/or copying this work, you (the licensee) agree that you
have read, understood, and will comply with the following terms and conditions.

Permission to copy, modify, and distribute this work, with or without
modification, for any purpose and without fee or royalty is hereby granted,
provided that you include the following on ALL copies of the work or portions
thereof, including modifications:

    - The full text of this NOTICE in a location viewable to users of the
      redistributed or derivative work.
    - Any pre-existing intellectual property disclaimers, notices, or terms and
      conditions. If none exist, the W3C software and document short notice
      should be included.
    - Notice of any changes or modifications, through a copyright statement on
      the new code or document such as "This software or document includes
      material copied from or derived from [title and URI of the W3C document].
      Copyright © [$year-of-document] World Wide Web Consortium.
      https://www.w3.org/copyright/software-license-2023/"

Disclaimers

THIS WORK IS PROVIDED "AS IS," AND COPYRIGHT HOLDERS MAKE NO REPRESENTATIONS
OR WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO, WARRANTIES OF
MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE
SOFTWARE OR DOCUMENT WILL NOT INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS,
TRADEMARKS OR OTHER RIGHTS.

COPYRIGHT HOLDERS WILL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL OR
CONSEQUENTIAL DAMAGES ARISING OUT OF ANY USE OF THE SOFTWARE OR DOCUMENT.

The name and trademarks of copyright holders may NOT be used in advertising or
publicity pertaining to the work without specific, written prior permission.
Title to copyright in this work will at all times remain with copyright holders.
"""
