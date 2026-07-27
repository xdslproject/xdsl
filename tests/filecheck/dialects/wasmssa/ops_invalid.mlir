// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%const = "wasmssa.const"() <{value = 1 : i128}> : () -> i128

// CHECK: Expected one of i32, i64, but got i128

// -----

%const = "wasmssa.const"() <{value = 1 : i32}> : () -> i64

// CHECK: attribute i64 expected from variable 'T', but got i32

// -----

%global = "wasmssa.global_get"() <{global = @module::@global}> : () -> i32

// CHECK: Expected SymbolRefAttr with no nested symbols.
