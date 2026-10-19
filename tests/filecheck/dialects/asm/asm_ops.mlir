// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%v, %r = "test.op"() : () -> (i32, !test.reg)

%from_reg_val = asm.from_reg %r {attr} : !test.reg -> i32
//         CHECK: %from_reg_val = asm.from_reg %r {attr} : !test.reg -> i32
// CHECK-GENERIC: %from_reg_val = "asm.from_reg"(%r) {attr} : (!test.reg) -> i32

%to_reg_val = asm.to_reg %v {attr} : i32 -> !test.reg
//         CHECK: %to_reg_val = asm.to_reg %v {attr} : i32 -> !test.reg
// CHECK-GENERIC: %to_reg_val = "asm.to_reg"(%v) {attr} : (i32) -> !test.reg

"test.op"(%from_reg_val, %to_reg_val) : (i32, !test.reg) -> ()
