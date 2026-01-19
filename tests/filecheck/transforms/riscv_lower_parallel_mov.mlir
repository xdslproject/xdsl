// RUN: xdsl-opt -p riscv-lower-parallel-mov --split-input-file --verify-diagnostics %s | filecheck %s

// Test no-op case
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s1>, !riscv.reg<s2>)
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
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s3>)
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
  %2, %3, %4 = riscv.parallel_mov %0, %1, %1 [32, 32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>)
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
  %3, %4, %5, %6 = riscv.parallel_mov %0, %1, %1, %2 [32, 32, 32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s2>, !riscv.reg<s5>) -> (!riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s6>)
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
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] {free_registers = [!riscv.reg<s10>]} : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
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

// Test cycle case when reusing register
//    s1
//   /  ^
//   |  |     s3 -> s4
//   v  /
//    s2
builtin.module {
  %0, %1, %2 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>)
  %3, %4, %5 = riscv.parallel_mov %0, %1, %2 [32, 32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>) -> (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s4>)
  "test.op"(%3, %4, %5) : (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s4>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>)
// CHECK-NEXT:    %3 = riscv.mv %2 : (!riscv.reg<s3>) -> !riscv.reg<s4>
// CHECK-NEXT:    %4 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s3>
// CHECK-NEXT:    %5 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s1>
// CHECK-NEXT:    %6 = riscv.mv %4 : (!riscv.reg<s3>) -> !riscv.reg<s2>
// CHECK-NEXT:    "test.op"(%6, %5, %3) : (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s4>) -> ()
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
  %5, %6, %7, %8, %9, %10, %11, %12, %13, %14 = riscv.parallel_mov %2, %0, %1, %0, %1, %2, %4, %3, %3, %3 [32, 32, 32, 32, 32, 32, 32, 32, 32, 32] {free_registers = [!riscv.reg<s11>]} : (!riscv.reg<s3>, !riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s8>, !riscv.reg<s7>, !riscv.reg<s7>, !riscv.reg<s7>) -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s7>, !riscv.reg<s8>, !riscv.reg<s9>, !riscv.reg<s10>)
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

// Test no free registers
//    s1
//   /  ^
//   |  |
//   v  /
//    s2
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg<s2>, !riscv.reg<s1>)
  "test.op"(%2, %3) : (!riscv.reg<s2>, !riscv.reg<s1>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    %2 = riscv.xor %1, %0 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %3 = riscv.xor %2, %0 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg<s1>
// CHECK-NEXT:    %4 = riscv.xor %2, %3 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    "test.op"(%4, %3) : (!riscv.reg<s2>, !riscv.reg<s1>) -> ()
// CHECK-NEXT:  }

// -----

// Test no free registers, multiple cycles
//    s1         s3 --> s4
//   /  ^        ^      |
//   |  |        |      |
//   v  /        |      v
//    s2         s6 <-- s5
builtin.module {
  %0, %1, %2, %3, %5, %6 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>)
  %7, %8, %9, %10, %11, %12 = riscv.parallel_mov %0, %1, %2, %3, %5, %6 [32, 32, 32, 32, 32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>) -> (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s3>)
  "test.op"(%7, %8, %9, %10, %11, %12) : (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s3>) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.reg<s3>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>)
// CHECK-NEXT:    %6 = riscv.xor %1, %0 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %7 = riscv.xor %6, %0 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg<s1>
// CHECK-NEXT:    %8 = riscv.xor %6, %7 : (!riscv.reg<s2>, !riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %9 = riscv.xor %5, %2 : (!riscv.reg<s6>, !riscv.reg<s3>) -> !riscv.reg<s6>
// CHECK-NEXT:    %10 = riscv.xor %9, %2 : (!riscv.reg<s6>, !riscv.reg<s3>) -> !riscv.reg<s3>
// CHECK-NEXT:    %11 = riscv.xor %9, %10 : (!riscv.reg<s6>, !riscv.reg<s3>) -> !riscv.reg<s6>
// CHECK-NEXT:    %12 = riscv.xor %4, %10 : (!riscv.reg<s5>, !riscv.reg<s3>) -> !riscv.reg<s5>
// CHECK-NEXT:    %13 = riscv.xor %12, %10 : (!riscv.reg<s5>, !riscv.reg<s3>) -> !riscv.reg<s3>
// CHECK-NEXT:    %14 = riscv.xor %12, %13 : (!riscv.reg<s5>, !riscv.reg<s3>) -> !riscv.reg<s5>
// CHECK-NEXT:    %15 = riscv.xor %3, %13 : (!riscv.reg<s4>, !riscv.reg<s3>) -> !riscv.reg<s4>
// CHECK-NEXT:    %16 = riscv.xor %15, %13 : (!riscv.reg<s4>, !riscv.reg<s3>) -> !riscv.reg<s3>
// CHECK-NEXT:    %17 = riscv.xor %15, %16 : (!riscv.reg<s4>, !riscv.reg<s3>) -> !riscv.reg<s4>
// CHECK-NEXT:    "test.op"(%8, %7, %17, %14, %11, %16) : (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.reg<s4>, !riscv.reg<s5>, !riscv.reg<s6>, !riscv.reg<s3>) -> ()
// CHECK-NEXT:  }

// -----

// Test moving floats and ints
// s1  -->  s2
// fs1 -->  fs2
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.freg<fs1>)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<s1>, !riscv.freg<fs1>) -> (!riscv.reg<s2>, !riscv.freg<fs2>)
  "test.op"(%2, %3) : (!riscv.reg<s2>, !riscv.freg<fs2>) -> ()
}
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.freg<fs1>)
// CHECK-NEXT:    %2 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s2>
// CHECK-NEXT:    %3 = riscv.fmv.s %1 : (!riscv.freg<fs1>) -> !riscv.freg<fs2>
// CHECK-NEXT:    "test.op"(%2, %3) : (!riscv.reg<s2>, !riscv.freg<fs2>) -> ()
// CHECK-NEXT:  }

// -----

// Test cyclic floats and ints
//    s1       fs1
//   /  ^     /  ^
//   |  |     |  |
//   v  /     v  /
//    s2       fs2
builtin.module {
  %0, %1, %2, %3 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.freg<fs1>, !riscv.freg<fs2>)
  %4, %5, %6, %7 = riscv.parallel_mov %0, %1, %2, %3 [32, 32, 32, 32] {free_registers = [!riscv.reg<s10>, !riscv.freg<fs10>]} : (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.freg<fs1>, !riscv.freg<fs2>) -> (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.freg<fs2>, !riscv.freg<fs1>)
  "test.op"(%4, %5, %6, %7) : (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.freg<fs2>, !riscv.freg<fs1>) -> ()
}
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>, !riscv.freg<fs1>, !riscv.freg<fs2>)
// CHECK-NEXT:    %4 = riscv.mv %0 : (!riscv.reg<s1>) -> !riscv.reg<s10>
// CHECK-NEXT:    %5 = riscv.mv %1 : (!riscv.reg<s2>) -> !riscv.reg<s1>
// CHECK-NEXT:    %6 = riscv.mv %4 : (!riscv.reg<s10>) -> !riscv.reg<s2>
// CHECK-NEXT:    %7 = riscv.fmv.s %2 : (!riscv.freg<fs1>) -> !riscv.freg<fs10>
// CHECK-NEXT:    %8 = riscv.fmv.s %3 : (!riscv.freg<fs2>) -> !riscv.freg<fs1>
// CHECK-NEXT:    %9 = riscv.fmv.s %7 : (!riscv.freg<fs10>) -> !riscv.freg<fs2>
// CHECK-NEXT:    "test.op"(%6, %5, %9, %8) : (!riscv.reg<s2>, !riscv.reg<s1>, !riscv.freg<fs2>, !riscv.freg<fs1>) -> ()
// CHECK-NEXT:  }

// -----

// Test different float types
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.freg<fs1>, !riscv.freg<fs2>)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 64] : (!riscv.freg<fs1>, !riscv.freg<fs2>) -> (!riscv.freg<fs3>, !riscv.freg<fs4>)
  "test.op"(%2, %3) : (!riscv.freg<fs3>, !riscv.freg<fs4>) -> ()
}
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!riscv.freg<fs1>, !riscv.freg<fs2>)
// CHECK-NEXT:    %2 = riscv.fmv.s %0 : (!riscv.freg<fs1>) -> !riscv.freg<fs3>
// CHECK-NEXT:    %3 = riscv.fmv.d %1 : (!riscv.freg<fs2>) -> !riscv.freg<fs4>
// CHECK-NEXT:    "test.op"(%2, %3) : (!riscv.freg<fs3>, !riscv.freg<fs4>) -> ()
// CHECK-NEXT:  }

// -----

// Test no free registers for float cycle
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.freg<fs1>, !riscv.freg<fs2>)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.freg<fs1>, !riscv.freg<fs2>) -> (!riscv.freg<fs2>, !riscv.freg<fs1>)
  "test.op"(%2, %3) : (!riscv.freg<fs2>, !riscv.freg<fs1>) -> ()
}

// CHECK:           %2, %3 = "riscv.parallel_mov"(%0, %1) <{input_widths = array<i32: 32, 32>}> : (!riscv.freg<fs1>, !riscv.freg<fs2>) -> (!riscv.freg<fs2>, !riscv.freg<fs1>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^------------------------------------------
// CHECK-NEXT:    | Error while applying pattern: Float cyclic move without free register
// CHECK-NEXT:    -----------------------------------------------------------------------

// -----

// Test unallocated inputs
builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg<s2>)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg, !riscv.reg<s2>)
}

// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) <{input_widths = array<i32: 32, 32>}> : (!riscv.reg<s1>, !riscv.reg<s2>) -> (!riscv.reg, !riscv.reg<s2>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------
// CHECK-NEXT:    | Error while applying pattern: All registers must be allocated
// CHECK-NEXT:    ---------------------------------------------------------------

// -----

builtin.module {
  %0, %1 = "test.op"() : () -> (!riscv.reg<s1>, !riscv.reg)
  %2, %3 = riscv.parallel_mov %0, %1 [32, 32] : (!riscv.reg<s1>, !riscv.reg) -> (!riscv.reg<s1>, !riscv.reg<s2>)
}
// CHECK:         %2, %3 = "riscv.parallel_mov"(%0, %1) <{input_widths = array<i32: 32, 32>}> : (!riscv.reg<s1>, !riscv.reg) -> (!riscv.reg<s1>, !riscv.reg<s2>)
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------
// CHECK-NEXT:    | Error while applying pattern: All registers must be allocated
// CHECK-NEXT:    ---------------------------------------------------------------
