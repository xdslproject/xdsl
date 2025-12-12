// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

builtin.module {
  %i0, %i1, %i2 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg)
  %o0 = riscv.mv %i0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
  %o1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
  %o2 = riscv.mv %i2 : (!riscv.reg) -> !riscv.reg
  "test.op"(%o0, %o1, %o2) : (!riscv.reg<a0>, !riscv.reg<a2>, !riscv.reg) -> ()

  %i3 = riscv.li 100 : !riscv.reg
  %i4 = riscv.mv %i3 : (!riscv.reg) -> !riscv.reg
  %i5 = riscv.mv %i3 : (!riscv.reg) -> !riscv.reg<j_0>
  "test.op"(%i3, %i4, %i5) : (!riscv.reg, !riscv.reg, !riscv.reg<j_0>) -> ()

  %f0, %f1, %f2 = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg)
  %fo0 = riscv.fmv.s %f0 : (!riscv.freg<fa0>) -> !riscv.freg<fa0>
  %fo1 = riscv.fmv.s %f1 : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
  %fo2 = riscv.fmv.s %f2 : (!riscv.freg) -> !riscv.freg
  %fo3 = riscv.fmv.d %f0 : (!riscv.freg<fa0>) -> !riscv.freg<fa0>
  %fo4 = riscv.fmv.d %f1 : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
  %fo5 = riscv.fmv.d %f2 : (!riscv.freg) -> !riscv.freg
  "test.op"(%fo0, %fo1, %fo2, %fo3, %fo4, %fo5) : (!riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg, !riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg) -> ()

  %zero = riscv.get_register : !riscv.reg<zero>
  %c0 = riscv.li 0 : !riscv.reg
  %c1 = riscv.li 1 : !riscv.reg
  %c2 = riscv.li 2 : !riscv.reg
  %c3 = riscv.li 3 : !riscv.reg

  // Don't optimise out unused immediates
  "test.op"(%zero, %c0, %c1, %c2, %c3) : (!riscv.reg<zero>, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

  %load_zero_zero = riscv.li 0 : !riscv.reg<zero>
  "test.op"(%load_zero_zero) : (!riscv.reg<zero>) -> ()

  %add_immediate_zero_reg = riscv.addi %zero, 1 : (!riscv.reg<zero>) -> !riscv.reg<a0>
  "test.op"(%add_immediate_zero_reg) : (!riscv.reg<a0>) -> ()

  %multiply_immediates = riscv.mul %c2, %c3 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%multiply_immediates) : (!riscv.reg<a0>) -> ()

  %multiply_immediate_r0 = riscv.mul %c0, %i1 : (!riscv.reg, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%multiply_immediate_r0) : (!riscv.reg<a0>) -> ()

  %multiply_immediate_l0 = riscv.mul %i1, %c0 : (!riscv.reg<a1>, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%multiply_immediate_l0) : (!riscv.reg<a0>) -> ()

  %multiply_immediate_r1 = riscv.mul %c1, %i1 : (!riscv.reg, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%multiply_immediate_r1) : (!riscv.reg<a0>) -> ()

  %multiply_immediate_l1 = riscv.mul %i1, %c1 : (!riscv.reg<a1>, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%multiply_immediate_l1) : (!riscv.reg<a0>) -> ()

  %div_lhs_const = riscv.div %i1, %c1 : (!riscv.reg<a1>, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%div_lhs_const) : (!riscv.reg<a0>) -> ()

  %add_lhs_immediate = riscv.add %c2, %i2 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%add_lhs_immediate) : (!riscv.reg<a0>) -> ()

  %add_rhs_immediate = riscv.add %i2, %c2 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%add_rhs_immediate) : (!riscv.reg<a0>) -> ()

  %add_immediates = riscv.add %c2, %c3 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%add_immediates) : (!riscv.reg<a0>) -> ()

  %add_vars = riscv.add %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

  %add_immediate_zero = riscv.addi %i2, 0 : (!riscv.reg) -> !riscv.reg<a0>
  "test.op"(%add_immediate_zero) : (!riscv.reg<a0>) -> ()

  %add_immediate_constant = riscv.addi %c2, 1 : (!riscv.reg) -> !riscv.reg<a0>
  "test.op"(%add_immediate_constant) : (!riscv.reg<a0>) -> ()

  // Unchanged
  %sub_lhs_immediate = riscv.sub %c2, %i2 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%sub_lhs_immediate) : (!riscv.reg<a0>) -> ()

  // Replace with addi
  %sub_rhs_immediate = riscv.sub %i2, %c2 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%sub_rhs_immediate) : (!riscv.reg<a0>) -> ()

  %sub_immediates = riscv.sub %c2, %c3 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%sub_immediates) : (!riscv.reg<a0>) -> ()

  %sub_lhs_rhs = riscv.sub %i1, %i1 : (!riscv.reg<a1>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%sub_lhs_rhs) : (!riscv.reg<a0>) -> ()

  // Unchanged
  %sub_vars = riscv.sub %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

  // Optimise out an arithmetic operation
  %sub_add_immediate = riscv.sub %add_rhs_immediate, %i2 : (!riscv.reg<a0>, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%sub_add_immediate) : (!riscv.reg<a0>) -> ()

  %andi_immediate = riscv.andi %i3, 7 : (!riscv.reg) -> !riscv.reg<a0>
  "test.op"(%andi_immediate) : (!riscv.reg<a0>) -> ()

  %shift_left_immediate = riscv.slli %c2, 4 : (!riscv.reg) -> !riscv.reg<a0>
  "test.op"(%shift_left_immediate) : (!riscv.reg<a0>) -> ()

  %load_float_ptr = riscv.addi %i2, 8 : (!riscv.reg) -> !riscv.reg
  %load_float_known_offset = riscv.flw %load_float_ptr, 4 : (!riscv.reg) -> !riscv.freg<fa0>
  "test.op"(%load_float_known_offset) : (!riscv.freg<fa0>) -> ()

  %load_double_ptr = riscv.addi %i2, 8 : (!riscv.reg) -> !riscv.reg
  %load_double_known_offset = riscv.fld %load_double_ptr, 4 : (!riscv.reg) -> !riscv.freg<fa0>
  "test.op"(%load_double_known_offset) : (!riscv.freg<fa0>) -> ()

  %store_float_ptr = riscv.addi %i2, 8 : (!riscv.reg) -> !riscv.reg
  riscv.fsw %store_float_ptr, %f2, 4 : (!riscv.reg, !riscv.freg) -> ()

  %store_double_ptr = riscv.addi %i2, 8 : (!riscv.reg) -> !riscv.reg
  riscv.fsd %store_double_ptr, %f2, 4 : (!riscv.reg, !riscv.freg) -> ()

  %add_lhs_rhs = riscv.add %i1, %i1 : (!riscv.reg<a1>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%add_lhs_rhs) : (!riscv.reg<a0>) -> ()

  %and_bitwise_zero_l0 = riscv.and %c1, %c0 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%and_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

  %and_bitwise_zero_r0 = riscv.and %c0, %c1 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%and_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

  %and_bitwise_self = riscv.and %i1, %i1 : (!riscv.reg<a1>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%and_bitwise_self) : (!riscv.reg<a0>) -> ()
  
  %or_bitwise_zero_l0 = riscv.or %c1, %c0 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%or_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

  %or_bitwise_zero_r0 = riscv.or %c0, %c1 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%or_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

  %or_bitwise_self = riscv.or %i1, %i1 : (!riscv.reg<a1>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%or_bitwise_self) : (!riscv.reg<a0>) -> ()

  %xor_lhs_rhs = riscv.xor %i1, %i1 : (!riscv.reg<a1>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%xor_lhs_rhs) : (!riscv.reg<a0>) -> ()

  %xor_bitwise_zero_l0 = riscv.xor %c1, %c0 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%xor_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

  %xor_bitwise_zero_r0 = riscv.xor %c0, %c1 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
  "test.op"(%xor_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

  %shift_left_zero_r0 = riscv.slli %i2, 0 : (!riscv.reg) -> !riscv.reg<a0>
  "test.op"(%shift_left_zero_r0) : (!riscv.reg<a0>) -> ()

  %shift_right_zero_r0 = riscv.srli %i2, 0 : (!riscv.reg) -> !riscv.reg<a0>
  "test.op"(%shift_right_zero_r0) : (!riscv.reg<a0>) -> ()

  // scfgw immediates
  riscv_snitch.scfgw %i1, %c1 : (!riscv.reg<a1>, !riscv.reg) -> ()
}

// CHECK: builtin.module {
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg)
// CHECK-NOT:    %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a1>) -> !riscv.reg<a2>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   "test.op"(%i0, %o1, %o2) : (!riscv.reg<a0>, !riscv.reg<a2>, !riscv.reg) -> ()

// CHECK-NEXT:   %i3 = riscv.li 100 : !riscv.reg
// CHECK-NEXT:   %i4 = riscv.mv %i3 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %i5 = riscv.mv %i3 : (!riscv.reg) -> !riscv.reg<j_0>
// CHECK-NEXT:   "test.op"(%i3, %i4, %i5) : (!riscv.reg, !riscv.reg, !riscv.reg<j_0>) -> ()

// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg)
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
// CHECK-NEXT:   %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:   "test.op"(%f0, %fo1, %fo2, %f0, %fo4, %fo5) : (!riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg, !riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg) -> ()

// CHECK-NEXT:   %zero = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %c0 = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %c0_1 = riscv.mv %c0 : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:   %c1 = riscv.li 1 : !riscv.reg
// CHECK-NEXT:   %c2 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:   %c3 = riscv.li 3 : !riscv.reg
// CHECK-NEXT:   "test.op"(%zero, %c0_1, %c1, %c2, %c3) : (!riscv.reg<zero>, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

// CHECK-NEXT:   %load_zero_zero = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:   "test.op"(%load_zero_zero) : (!riscv.reg<zero>) -> ()

// CHECK-NEXT:   %add_immediate_zero_reg = riscv.li 1 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediate_zero_reg) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediates = riscv.li 6 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediate_r0 = riscv.mv %c0_1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediate_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediate_l0 = riscv.mv %c0_1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediate_l0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediate_r1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediate_r1) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediate_l1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediate_l1) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %div_lhs_const = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%div_lhs_const) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_lhs_immediate = riscv.addi %i2, 2 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_lhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_rhs_immediate = riscv.addi %i2, 2 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_rhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediates = riscv.li 5 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_vars = riscv.add %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediate_zero = riscv.mv %i2 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediate_zero) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediate_constant = riscv.li 3 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediate_constant) : (!riscv.reg<a0>) -> ()

  // Unchanged
// CHECK-NEXT:   %sub_lhs_immediate = riscv.sub %c2, %i2 : (!riscv.reg, !riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_lhs_immediate) : (!riscv.reg<a0>) -> ()

  // Replace with addi
// CHECK-NEXT:   %sub_rhs_immediate = riscv.addi %i2, -2 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_rhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %sub_immediates = riscv.li -1 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %sub_lhs_rhs = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %sub_lhs_rhs_1 = riscv.mv %sub_lhs_rhs : (!riscv.reg<zero>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_lhs_rhs_1) : (!riscv.reg<a0>) -> ()

  // Unchanged
// CHECK-NEXT:   %sub_vars = riscv.sub %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

// Optimise out an arithmetic operation
// CHECK-NEXT:   %sub_add_immediate = riscv.li 2 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_add_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %andi_immediate = riscv.li 4 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%andi_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %shift_left_immediate = riscv.li 32 : !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%shift_left_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %load_float_known_offset = riscv.flw %i2, 12 : (!riscv.reg) -> !riscv.freg<fa0>
// CHECK-NEXT:   "test.op"(%load_float_known_offset) : (!riscv.freg<fa0>) -> ()

// CHECK-NEXT:   %load_double_known_offset = riscv.fld %i2, 12 : (!riscv.reg) -> !riscv.freg<fa0>
// CHECK-NEXT:   "test.op"(%load_double_known_offset) : (!riscv.freg<fa0>) -> ()

// CHECK-NEXT:   riscv.fsw %i2, %f2, 12 : (!riscv.reg, !riscv.freg) -> ()

// CHECK-NEXT:   riscv.fsd %i2, %f2, 12 : (!riscv.reg, !riscv.freg) -> ()

// CHECK-NEXT:   %add_lhs_rhs = riscv.li 2 : !riscv.reg
// CHECK-NEXT:   %add_lhs_rhs_1 = riscv.mul %i1, %add_lhs_rhs : (!riscv.reg<a1>, !riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_lhs_rhs_1) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %and_bitwise_zero_l0 = riscv.mv %c0_1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%and_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %and_bitwise_zero_r0 = riscv.mv %c0_1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%and_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %and_bitwise_self = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%and_bitwise_self) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %or_bitwise_zero_l0 = riscv.mv %c1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%or_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %or_bitwise_zero_r0 = riscv.mv %c1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%or_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %or_bitwise_self = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%or_bitwise_self) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %xor_lhs_rhs = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:   %xor_lhs_rhs_1 = riscv.mv %xor_lhs_rhs : (!riscv.reg<zero>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%xor_lhs_rhs_1) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %xor_bitwise_zero_l0 = riscv.mv %c1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%xor_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %xor_bitwise_zero_r0 = riscv.mv %c1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%xor_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %shift_left_zero_r0 = riscv.mv %i2 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%shift_left_zero_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %shift_right_zero_r0 = riscv.mv %i2 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%shift_right_zero_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   riscv_snitch.scfgwi %i1, 1 : (!riscv.reg<a1>) -> ()

// CHECK-NEXT: }

// -----

%0, %1 = "test.op"() : () -> (!riscv.freg, !riscv.freg)

// should fuse
%rmul0 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd0 = riscv.fadd.d %0, %rmul0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// same as above, but swapped addends
%rmul0b = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd0b = riscv.fadd.d %rmul0b, %0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// same as above but allocated
%rmul0_a = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd0_a = riscv.fadd.d %0, %rmul0_a fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg<ft0>

%rmul0b_a = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd0b_a = riscv.fadd.d %rmul0b_a, %0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg<ft1>

// both addends are results of multiplcation, if all else is the same we fuse with second operand
%rmul0c0 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%rmul0c1 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd0c = riscv.fadd.d %rmul0c0, %rmul0c1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// should not fuse due to missing "contract" fastmath flag
%rmul1 = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd1 = riscv.fadd.d %0, %rmul1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

%rmul1b = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd1b = riscv.fadd.d %rmul1b, %0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// should not fuse due to missing "contract" fastmath flag
%rmul2 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd2 = riscv.fadd.d %0, %rmul2 : (!riscv.freg, !riscv.freg) -> !riscv.freg

%rmul2b = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd2b = riscv.fadd.d %rmul2b, %0 : (!riscv.freg, !riscv.freg) -> !riscv.freg

// should not fuse due to missing "contract" fastmath flag
%rmul3 = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd3 = riscv.fadd.d %0, %rmul3 : (!riscv.freg, !riscv.freg) -> !riscv.freg

%rmul3b = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd3b = riscv.fadd.d %rmul3b, %0 : (!riscv.freg, !riscv.freg) -> !riscv.freg

// should not fuse due to more than one uses
%rmul4 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd4 = riscv.fadd.d %0, %rmul4 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

%rmul4b = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%radd4b = riscv.fadd.d %rmul4b, %0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// use multiplication result here to stop the fusion
"test.op"(%rmul4) : (!riscv.freg) -> ()
"test.op"(%rmul4b) : (!riscv.freg) -> ()

// use results here to avoid dead code elimination up the SSA chain
"test.op"(%radd0) : (!riscv.freg) -> ()
"test.op"(%radd0b) : (!riscv.freg) -> ()
"test.op"(%radd0_a) : (!riscv.freg<ft0>) -> ()
"test.op"(%radd0b_a) : (!riscv.freg<ft1>) -> ()
"test.op"(%radd0c) : (!riscv.freg) -> ()
"test.op"(%radd1) : (!riscv.freg) -> ()
"test.op"(%radd1b) : (!riscv.freg) -> ()
"test.op"(%radd2) : (!riscv.freg) -> ()
"test.op"(%radd2b) : (!riscv.freg) -> ()
"test.op"(%radd3) : (!riscv.freg) -> ()
"test.op"(%radd3b) : (!riscv.freg) -> ()
"test.op"(%radd4) : (!riscv.freg) -> ()
"test.op"(%radd4b) : (!riscv.freg) -> ()

// CHECK:      builtin.module {

// CHECK-NEXT:   %0, %1 = "test.op"() : () -> (!riscv.freg, !riscv.freg)

// CHECK-NEXT:   %radd0 = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %radd0b = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %radd0_a = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg<ft0>

// CHECK-NEXT:   %radd0b_a = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg<ft1>

// CHECK-NEXT:   %rmul0c0 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd0c = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul1 = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd1 = riscv.fadd.d %0, %rmul1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul1b = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd1b = riscv.fadd.d %rmul1b, %0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul2 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd2 = riscv.fadd.d %0, %rmul2 : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul2b = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd2b = riscv.fadd.d %rmul2b, %0 : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul3 = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd3 = riscv.fadd.d %0, %rmul3 : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul3b = riscv.fmul.d %0, %1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd3b = riscv.fadd.d %rmul3b, %0 : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul4 = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd4 = riscv.fadd.d %0, %rmul4 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   %rmul4b = riscv.fmul.d %0, %1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %radd4b = riscv.fadd.d %rmul4b, %0 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

// CHECK-NEXT:   "test.op"(%rmul4) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%rmul4b) : (!riscv.freg) -> ()

// CHECK-NEXT:   "test.op"(%radd0) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd0b) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd0_a) : (!riscv.freg<ft0>) -> ()
// CHECK-NEXT:   "test.op"(%radd0b_a) : (!riscv.freg<ft1>) -> ()
// CHECK-NEXT:   "test.op"(%radd0c) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd1) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd1b) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd2) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd2b) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd3) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd3b) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd4) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%radd4b) : (!riscv.freg) -> ()

// CHECK-NEXT:  }
