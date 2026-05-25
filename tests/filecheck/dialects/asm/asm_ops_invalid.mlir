// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

%v = "test.op"() : () -> i32
%v1 = asm.from_reg %v {attr} : i32 -> i64

// CHECK: i32 should be of attribute subclassing `RegisterType`

// -----
%v = "test.op"() : () -> i32
%v1 = asm.to_reg %v {attr} : i32 -> i64

// CHECK: i64 should be of attribute subclassing `RegisterType`
