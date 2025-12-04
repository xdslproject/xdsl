// RUN: xdsl-opt -p riscv-lower-parallel-mov --split-input-file --verify-diagnostics %s | filecheck %s

// Test no-op case
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %4 = riscv.add %2, %3 : (!riscv.reg<s1>, !riscv.reg<s2>) -> !riscv.reg
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.add %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> !riscv.reg
// CHECK-NEXT:  }

// -----

// Test non-cycle case
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s3>)
  %4 = riscv.add %2, %3 : (!riscv.reg<s2>, !riscv.reg<s3>) -> !riscv.reg
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s3>
// CHECK-NEXT:    %3 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %4 = riscv.add %3, %2 : (!riscv.reg<s2>, !riscv.reg<s3>) -> !riscv.reg
// CHECK-NEXT:  }
// -----

// Test cycle case
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
  %4 = riscv.add %2, %3 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg
}

// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^------------------
// CHECK-NEXT:    | Error while applying pattern: Not implemented
// CHECK-NEXT:    -----------------------------------------------

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
