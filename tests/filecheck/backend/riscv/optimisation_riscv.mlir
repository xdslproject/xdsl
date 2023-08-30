// RUN: xdsl-opt -p canonicalize %s | filecheck %s

builtin.module {
  %i0, %i1, %i2 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<>)
  %o0 = riscv.mv %i0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
  %o1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
  %o2 = riscv.mv %i2 : (!riscv.reg<>) -> !riscv.reg<>

  %f0, %f1, %f2 = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg<>)
  %fo0 = riscv.fmv.s %f0 : (!riscv.freg<fa0>) -> !riscv.freg<fa0>
  %fo1 = riscv.fmv.s %f1 : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
  %fo2 = riscv.fmv.s %f2 : (!riscv.freg<>) -> !riscv.freg<>

  %0 = riscv.li 0 : () -> !riscv.reg<>
  %1 = riscv.li 1 : () -> !riscv.reg<>
  %2 = riscv.li 2 : () -> !riscv.reg<>
  %3 = riscv.li 3 : () -> !riscv.reg<>

  // Don't optimise out unused immediates
  "test.op"(%0, %1, %2, %3) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

  %multiply_immediates = riscv.mul %2, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%multiply_immediates) : (!riscv.reg<a0>) -> ()

  %add_lhs_immediate = riscv.add %2, %i2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_lhs_immediate) : (!riscv.reg<a0>) -> ()

  %add_rhs_immediate = riscv.add %i2, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_rhs_immediate) : (!riscv.reg<a0>) -> ()

  %add_immediates = riscv.add %2, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_immediates) : (!riscv.reg<a0>) -> ()

  %shift_left_immediate = riscv.slli %2, 4 : (!riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%shift_left_immediate) : (!riscv.reg<a0>) -> ()

  %load_float_ptr = riscv.addi %i2, 8 : (!riscv.reg<>) -> !riscv.reg<>
  %load_float_known_offset = riscv.flw %load_float_ptr, 4 : (!riscv.reg<>) -> !riscv.freg<fa0>
  "test.op"(%load_float_known_offset) : (!riscv.freg<fa0>) -> ()

  %store_float_ptr = riscv.addi %i2, 8 : (!riscv.reg<>) -> !riscv.reg<>
  riscv.fsw %store_float_ptr, %f2, 4 : (!riscv.reg<>, !riscv.freg<>) -> ()
}

// CHECK: builtin.module {
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<>)
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a1>) -> !riscv.reg<a2>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg<>)
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg<>) -> !riscv.freg<>

// CHECK-NEXT:   %0 = riscv.get_zero_register : () -> !riscv.reg<zero>
// CHECK-NEXT:   %1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:   %2 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %3 = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:   "test.op"(%0, %1, %2, %3) : (!riscv.reg<zero>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

// CHECK-NEXT:   %multiply_immediates = riscv.li 6 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_lhs_immediate = riscv.addi %i2, 2 : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_lhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_rhs_immediate = riscv.addi %i2, 2 : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_rhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediates = riscv.li 5 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %shift_left_immediate = riscv.li 32 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%shift_left_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %load_float_known_offset = riscv.flw %i2, 12 : (!riscv.reg<>) -> !riscv.freg<fa0>
// CHECK-NEXT:   "test.op"(%load_float_known_offset) : (!riscv.freg<fa0>) -> ()

// CHECK-NEXT:   riscv.fsw %i2, %f2, 12 : (!riscv.reg<>, !riscv.freg<>) -> ()

// CHECK-NEXT: }
