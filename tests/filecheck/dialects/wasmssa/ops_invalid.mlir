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

// -----

%lhs, %rhs = "test.op"() : () -> (f32, f32)
%result = "wasmssa.div_ui"(%lhs, %rhs) : (f32, f32) -> f32

// CHECK: Expected one of i32, i64, but got f32

// -----

%lhs, %rhs = "test.op"() : () -> (i32, i32)
%result = "wasmssa.div"(%lhs, %rhs) : (i32, i32) -> i32

// CHECK: Unexpected attribute i32

// -----

%lhs = "test.op"() : () -> i32
%rhs = "test.op"() : () -> i64
%comparison = "wasmssa.eq"(%lhs, %rhs) : (i32, i64) -> i32

// CHECK: attribute i32 expected from variable 'T', but got i64

// -----

%lhs = "test.op"() : () -> i32
%rhs = "test.op"() : () -> i32
%comparison = "wasmssa.lt"(%lhs, %rhs) : (i32, i32) -> i32

// CHECK: Unexpected attribute i32

// -----

%lhs = "test.op"() : () -> f32
%rhs = "test.op"() : () -> f32
%comparison = "wasmssa.lt_si"(%lhs, %rhs) : (f32, f32) -> i32

// CHECK: Expected one of i32, i64, but got f32

// -----

%input = "test.op"() : () -> f32
%comparison = "wasmssa.eqz"(%input) : (f32) -> i32

// CHECK: Expected one of i32, i64, but got f32

// -----

%lhs = "test.op"() : () -> i32
%rhs = "test.op"() : () -> i32
%comparison = "wasmssa.ne"(%lhs, %rhs) : (i32, i32) -> i64

// CHECK: Expected attribute i32 but got i64

// -----

%src = "test.op"() : () -> i32
%result = "wasmssa.abs"(%src) : (i32) -> i32

// CHECK: Unexpected attribute i32

// -----

%src = "test.op"() : () -> f32
%result = "wasmssa.clz"(%src) : (f32) -> f32

// CHECK: Expected one of i32, i64, but got f32

// -----

%input = "test.op"() : () -> f32
%result = "wasmssa.convert_s"(%input) : (f32) -> f64

// CHECK: Expected one of i32, i64, but got f32

// -----

%src = "test.op"() : () -> f32
%result = "wasmssa.sqrt"(%src) : (f32) -> f64

// CHECK: attribute f32 expected from variable 'T', but got f64

// -----

%input = "test.op"() : () -> i32
%result = "wasmssa.convert_u"(%input) : (i32) -> i64

// CHECK: Unexpected attribute i64

// -----

%input = "test.op"() : () -> f32
%result = "wasmssa.demote"(%input) : (f32) -> f32

// CHECK: f32 should be of base attribute f64

// -----

%input = "test.op"() : () -> f32
%result = "wasmssa.extend_i32_s"(%input) : (f32) -> i64

// CHECK: Expected attribute i32 but got f32

// -----

%input = "test.op"() : () -> i32
%result = wasmssa.extend 67 : i64 low bits from %input : i32

// CHECK: extend op can only take 8, 16 or 32 bits. Got 67

// -----

%input = "test.op"() : () -> i32
%result = wasmssa.extend 32 : i64 low bits from %input : i32

// CHECK: trying to extend the 32 low bits from a i32 value is illegal

// -----

%input = "test.op"() : () -> i32
%result = "wasmssa.extend"(%input) <{bitsToTake = 8 : i64}> : (i32) -> i64

// CHECK: attribute i32 expected from variable 'T', but got i64

// -----

%input = "test.op"() : () -> i32
%result = wasmssa.reinterpret %input : i32 as i32

// CHECK: reinterpret input and output type should be distinct

// -----

%input = "test.op"() : () -> i32
%result = wasmssa.reinterpret %input : i32 as f64

// CHECK: input type (i32) and output type (f64) have incompatible bit widths
