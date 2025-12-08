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

// Test tree case:
//    s1
//    |
//    v
//    s2
//   /  \
//  v    v
// s3    s4
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3, %4 = riscv.parallel_mov %0, %1, %1 : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>)
  "test.op"(%2, %3, %4) : (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s3>
// CHECK-NEXT:    %3 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s4>
// CHECK-NEXT:    %4 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    "test.op"(%4, %2, %3) : (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>) -> ()
// CHECK-NEXT:  }
// -----

// Test two trees case:
//    s1            s5
//    |             |
//    v             v
//    s2            s6
//   /  \
//  v    v
// s3    s4
builtin.module {
  %0, %1, %2 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s5>)
  %3, %4, %5, %6 = riscv.parallel_mov %0, %1, %1, %2 : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s2>, !riscv.reg<s5>) -> (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s6>)
  "test.op"(%3, %4, %5, %6) : (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s6>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s5>)
// CHECK-NEXT:    %3 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s3>
// CHECK-NEXT:    %4 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s4>
// CHECK-NEXT:    %5 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %6 = riscv.mv %2 : (!riscv.reg<s5>) -> !riscv.reg<s6>
// CHECK-NEXT:    "test.op"(%5, %3, %4, %6) : (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s6>) -> ()
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
  %2, %3 = riscv.parallel_mov %0, %1 {free_registers = [!riscv.reg<s10>]} : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
  "test.op"(%2, %3) : (!riscv.reg<s2>, !riscv.reg<s1>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s10>
// CHECK-NEXT:    %3 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s1>
// CHECK-NEXT:    %4 = riscv.mv %2 : (!riscv.reg<s10>) -> !riscv.reg<s2>
// CHECK-NEXT:    "test.op"(%4, %3) : (!riscv.reg<s2>, !riscv.reg<s1>) -> ()
// CHECK-NEXT:  }

// -----

// Test complex case
//     s4
//     ^
//     |
//     s1               --> s8
//    / ^              /    /
//   v   \            s7 <--
//  s2 -> s3         /  \
//  |     |         v    v
//  v     v        s9    s10
//  s5    s6
//
builtin.module {
  %0, %1, %2, %3, %4 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s7>, !riscv.reg<s8>)
  %5, %6, %7, %8, %9, %10, %11, %12, %13, %14 = riscv.parallel_mov %2, %0, %1, %0, %1, %2, %4, %3, %3, %3 {free_registers = [!riscv.reg<s11>]} : (!riscv.reg<s3>, !riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s8>, !riscv.reg<s7>, !riscv.reg<s7>, !riscv.reg<s7>) -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s7>, !riscv.reg<s8>, !riscv.reg<s9>, !riscv.reg<s10>)
  "test.op"(%5, %6, %7, %8, %9, %10, %11, %12, %13, %14) : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s7>, !riscv.reg<s8>, !riscv.reg<s9>, !riscv.reg<s10>) -> ()
}
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3, %4 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s7>, !riscv.reg<s8>)
// CHECK-NEXT:    %5 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s4>
// CHECK-NEXT:    %6 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s5>
// CHECK-NEXT:    %7 = riscv.mv %2 : (!riscv.reg<s3>) -> !riscv.reg<s6>
// CHECK-NEXT:    %8 = riscv.mv %3 : (!riscv.reg<s7>) -> !riscv.reg<s9>
// CHECK-NEXT:    %9 = riscv.mv %3 : (!riscv.reg<s7>) -> !riscv.reg<s10>
// CHECK-NEXT:    %10 = riscv.mv %2 : (!riscv.reg<s3>) -> !riscv.reg<s11>
// CHECK-NEXT:    %11 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s3>
// CHECK-NEXT:    %12 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %13 = riscv.mv %10 : (!riscv.reg<s11>) -> !riscv.reg<s1>
// CHECK-NEXT:    %14 = riscv.mv %4 : (!riscv.reg<s8>) -> !riscv.reg<s11>
// CHECK-NEXT:    %15 = riscv.mv %3 : (!riscv.reg<s7>) -> !riscv.reg<s8>
// CHECK-NEXT:    %16 = riscv.mv %14 : (!riscv.reg<s11>) -> !riscv.reg<s7>
// CHECK-NEXT:    "test.op"(%13, %12, %11, %5, %6, %7, %16, %15, %8, %9) : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s7>, !riscv.reg<s8>, !riscv.reg<s9>, !riscv.reg<s10>) -> ()
// CHECK-NEXT:  }

// -----

// Test moving floats and ints
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.freg<ft1>)
  %2, %3 = riscv.parallel_mov %0, %1 : (!riscv.reg<s1>, !riscv.freg<ft1>) -> (!riscv.reg<s2>, !riscv.freg<ft2>)
}
// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) : (!riscv.reg<s1>, !riscv.freg<ft1>) -> (!riscv.reg<s2>, !riscv.freg<ft2>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------------
// CHECK-NEXT:    | Error while applying pattern: Not implemented: only integer support
// CHECK-NEXT:    ---------------------------------------------------------------------

// -----

// Test no free registers
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
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-----------------------------------------------
// CHECK-NEXT:    | Error while applying pattern: Cyclic move detected with no free registers.
// CHECK-NEXT:    ----------------------------------------------------------------------------

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
