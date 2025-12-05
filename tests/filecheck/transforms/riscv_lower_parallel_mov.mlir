// RUN: xdsl-opt -p riscv-lower-parallel-mov --split-input-file --verify-diagnostics %s | filecheck %s

// Test no-op case
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s1>, !riscv.reg<s2>)
  "test.op"(%2, %3) : (!riscv.reg<s1>, !riscv.reg<s2>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    "test.op"(%0, %1) : (!riscv.reg<s1>, !riscv.reg<s2>) -> ()
// CHECK-NEXT:  }

// -----

// Test chain case:
//   s1
//   |
//   v
//   s2
//   |
//   v
//   s3
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s3>)
  "test.op"(%2, %3) : (!riscv.reg<s2>, !riscv.reg<s3>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s3>
// CHECK-NEXT:    %3 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    "test.op"(%3, %2) : (!riscv.reg<s2>, !riscv.reg<s3>)
// CHECK-NEXT:  }
// -----

// Test cycle case
//    s1
//   /  ^
//   |  |
//   v  /
//    s2
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
  "test.op"(%2, %3) : (!riscv.reg<s2>, !riscv.reg<s1>) -> ()
}

// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^--------------------------------
// CHECK-NEXT:    | Error while applying pattern: Not implemented: cyclic moves
// CHECK-NEXT:    -------------------------------------------------------------

// -----

// Test unallocated inputs
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg, !riscv.reg<s2>)
}

// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg, !riscv.reg<s2>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------
// CHECK-NEXT:    | Error while applying pattern: All registers must be allocated
// CHECK-NEXT:    ---------------------------------------------------------------

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg) -> (!riscv.reg<s1>, !riscv.reg<s2>)
}
// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) : (!riscv.reg<s1>, !riscv.reg) -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------
// CHECK-NEXT:    | Error while applying pattern: All registers must be allocated
// CHECK-NEXT:    ---------------------------------------------------------------
