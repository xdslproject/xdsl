// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%v, %r = "test.op"() : () -> (i32, !test.reg)

//         CHECK: %from_reg_val = asm.from_reg %r {attr} : !test.reg -> i32
// CHECK-GENERIC: %from_reg_val = "asm.from_reg"(%r) {attr} : (!test.reg) -> i32
%from_reg_val = asm.from_reg %r {attr} : !test.reg -> i32

//         CHECK: %to_reg_val = asm.to_reg %v {attr} : i32 -> !test.reg
// CHECK-GENERIC: %to_reg_val = "asm.to_reg"(%v) {attr} : (i32) -> !test.reg
%to_reg_val = asm.to_reg %v {attr} : i32 -> !test.reg

"test.op"(%from_reg_val, %to_reg_val) : (i32, !test.reg) -> ()

//      CHECK: asm.region {
// CHECK-NEXT:     asm.yield
// CHECK-NEXT: } : () -> ()
asm.region {
    asm.yield
} : () -> ()

//      CHECK: %res_v = asm.region(%v) {
// CHECK-NEXT: ^bb0(%arg_r: !test.reg):
// CHECK-NEXT:     %res_r = "test.op"(%arg_r) : (!test.reg) -> !test.reg
// CHECK-NEXT:     asm.yield %res_r : !test.reg
// CHECK-NEXT: } : (i32) -> i32
%res_v = asm.region(%v) {
^bb0(%arg_r: !test.reg):
    %res_r = "test.op"(%arg_r) : (!test.reg) -> !test.reg
    asm.yield %res_r : !test.reg
} : (i32) -> i32

//      CHECK: %res_v0, %res_v1 = asm.region(%v, %res_v) {
// CHECK-NEXT: ^bbo(%arg_r0: !test.reg, %arg_r1: !test.reg):
// CHECK-NEXT:     %res_r0, %res_r1 = "test.op"(%arg_r0, %arg_r1) : (!test.reg, !test.reg) -> (!test.reg, !test.reg)
// CHECK-NEXT:     asm.yield %res_r0, %res_r1 : !test.reg, !test.reg
// CHECK-NEXT: } : (i32, i32) -> (i32, i32)
%res_v0, %res_v1 = asm.region(%v, %res_v) {
^bbo(%arg_r0: !test.reg, %arg_r1: !test.reg):
    %res_r0, %res_r1 = "test.op"(%arg_r0, %arg_r1) : (!test.reg, !test.reg) -> (!test.reg, !test.reg)
    asm.yield %res_r0, %res_r1 : !test.reg, !test.reg
} : (i32, i32) -> (i32, i32)
