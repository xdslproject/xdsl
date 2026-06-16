// RUN: xdsl-opt -p canonicalize %s | filecheck %s

%r0 = "test.op"() : () -> !test.reg
%v = asm.from_reg %r0 : !test.reg -> i32
%r1 = asm.to_reg %v: i32 -> !test.reg
"test.op"(%r1) : (!test.reg) -> ()

//      CHECK: %r0 = "test.op"() : () -> !test.reg
// CHECK-NEXT: "test.op"(%r0) : (!test.reg) -> ()

%v0 = "test.op"() : () -> i32
%r = asm.to_reg %v0 : i32 -> !test.reg
%v1 = asm.from_reg %r : !test.reg -> i32
"test.op"(%v1) : (i32) -> ()

//      CHECK: %v0 = "test.op"() : () -> i32
// CHECK-NEXT: "test.op"(%v0) : (i32) -> ()
