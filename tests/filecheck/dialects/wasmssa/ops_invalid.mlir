// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%const = "wasmssa.const"() <{value = 1 : i128}> : () -> i128

// CHECK: Expected one of i32, i64, but got i128

// -----

%const = "wasmssa.const"() <{value = 1 : i32}> : () -> i64

// CHECK: attribute i64 expected from variable 'T', but got i32

// -----

%global = "wasmssa.global_get"() <{global = @module::@global}> : () -> i32

// CHECK: Expected SymbolRefAttr with no nested symbols.

// -----

%lhs = "test.op"() : () -> i128
%rhs = "test.op"() : () -> i128
%sum = "wasmssa.add"(%lhs, %rhs) : (i128, i128) -> i128

// CHECK: Expected one of i32, i64, but got i128

// -----

%lhs = "test.op"() : () -> i32
%rhs = "test.op"() : () -> i64
%sum = "wasmssa.add"(%lhs, %rhs) : (i32, i64) -> i32

// CHECK: attribute i32 expected from variable 'T', but got i64
