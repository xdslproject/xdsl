// RUN: xdsl-opt %s -p dce | filecheck %s

"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"() : () -> i32

  %a = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %b = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %c = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %d = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %e = "arith.addi"(%0, %1) : (i32, i32) -> i32

  "test.op"(%0) : (i32) -> ()

  // CHECK:       %0 = "test.op"() : () -> i32
  // CHECK-NEXT:  %1 = "test.op"() : () -> i32
  // CHECK-NEXT:  "test.op"(%0) : (i32) -> ()
}) : () -> ()
