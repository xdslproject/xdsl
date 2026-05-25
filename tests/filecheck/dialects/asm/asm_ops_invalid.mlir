// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

%v = "test.op"() : () -> i32
%v1 = asm.from_reg %v {attr} : i32 -> i64

// CHECK: i32 should be of attribute subclassing `RegisterType`

// -----
%v = "test.op"() : () -> i32
%v1 = asm.to_reg %v {attr} : i32 -> i64

// CHECK: i64 should be of attribute subclassing `RegisterType`

// -----

%v0 = "test.op"() : () -> i32
%r = asm.to_reg %v0 {attr} : i32 -> !test.reg
%v1 = asm.from_reg %r {attr} : !test.reg -> i64

// CHECK: Expected original value type i32 to be equal to own value type i64.

// -----

%r = "test.op"() : () -> !test.reg<x0>
%v = asm.from_reg %r {attr} : !test.reg<x0> -> i32
%r2 = asm.to_reg %v {attr} : i32 -> !test.reg<x1>

// CHECK: Expected original register type !test.reg<x0> to be equal to own register type !test.reg<x1>.
