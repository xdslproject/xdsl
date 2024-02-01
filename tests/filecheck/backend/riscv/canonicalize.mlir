// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

builtin.module {
  %i0, %i1, %i2 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<>)
  %o0 = riscv.mv %i0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
  %o1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
  %o2 = riscv.mv %i2 : (!riscv.reg<>) -> !riscv.reg<>
  "test.op"(%o0, %o1, %o2) : (!riscv.reg<a0>, !riscv.reg<a2>, !riscv.reg<>) -> ()

  %i3 = riscv.li 100 : () -> !riscv.reg<>
  %i4 = riscv.mv %i3 : (!riscv.reg<>) -> !riscv.reg<>
  %i5 = riscv.mv %i3 : (!riscv.reg<>) -> !riscv.reg<j0>
  "test.op"(%i3, %i4, %i5) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<j0>) -> ()

  %f0, %f1, %f2 = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg<>)
  %fo0 = riscv.fmv.s %f0 : (!riscv.freg<fa0>) -> !riscv.freg<fa0>
  %fo1 = riscv.fmv.s %f1 : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
  %fo2 = riscv.fmv.s %f2 : (!riscv.freg<>) -> !riscv.freg<>
  %fo3 = riscv.fmv.d %f0 : (!riscv.freg<fa0>) -> !riscv.freg<fa0>
  %fo4 = riscv.fmv.d %f1 : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
  %fo5 = riscv.fmv.d %f2 : (!riscv.freg<>) -> !riscv.freg<>
  "test.op"(%fo0, %fo1, %fo2, %fo3, %fo4, %fo5) : (!riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg<>, !riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg<>) -> ()

  %0 = riscv.li 0 : () -> !riscv.reg<>
  %1 = riscv.li 1 : () -> !riscv.reg<>
  %2 = riscv.li 2 : () -> !riscv.reg<>
  %3 = riscv.li 3 : () -> !riscv.reg<>

  // Don't optimise out unused immediates
  "test.op"(%0, %1, %2, %3) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

  %multiply_immediates = riscv.mul %2, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%multiply_immediates) : (!riscv.reg<a0>) -> ()

  %multiply_immediate_r0 = riscv.mul %0, %i1 : (!riscv.reg<>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%multiply_immediate_r0) : (!riscv.reg<a0>) -> ()

  %multiply_immediate_l0 = riscv.mul %i1, %0 : (!riscv.reg<a1>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%multiply_immediate_l0) : (!riscv.reg<a0>) -> ()

  %add_lhs_immediate = riscv.add %2, %i2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_lhs_immediate) : (!riscv.reg<a0>) -> ()

  %add_rhs_immediate = riscv.add %i2, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_rhs_immediate) : (!riscv.reg<a0>) -> ()

  %add_immediates = riscv.add %2, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_immediates) : (!riscv.reg<a0>) -> ()

  %add_vars = riscv.add %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

  %add_immediate_zero = riscv.addi %i2, 0 : (!riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_immediate_zero) : (!riscv.reg<a0>) -> ()

  %add_immediate_constant = riscv.addi %2, 1 : (!riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%add_immediate_constant) : (!riscv.reg<a0>) -> ()

  // Unchanged
  %sub_lhs_immediate = riscv.sub %2, %i2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%sub_lhs_immediate) : (!riscv.reg<a0>) -> ()

  // Replace with addi
  %sub_rhs_immediate = riscv.sub %i2, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%sub_rhs_immediate) : (!riscv.reg<a0>) -> ()

  %sub_immediates = riscv.sub %2, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%sub_immediates) : (!riscv.reg<a0>) -> ()

  // Unchanged
  %sub_vars = riscv.sub %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

  // Optimise out an arithmetic operation
  %sub_add_immediate = riscv.sub %add_rhs_immediate, %i2 : (!riscv.reg<a0>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%sub_add_immediate) : (!riscv.reg<a0>) -> ()

  %shift_left_immediate = riscv.slli %2, 4 : (!riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%shift_left_immediate) : (!riscv.reg<a0>) -> ()

  %load_float_ptr = riscv.addi %i2, 8 : (!riscv.reg<>) -> !riscv.reg<>
  %load_float_known_offset = riscv.flw %load_float_ptr, 4 : (!riscv.reg<>) -> !riscv.freg<fa0>
  "test.op"(%load_float_known_offset) : (!riscv.freg<fa0>) -> ()

  %load_double_ptr = riscv.addi %i2, 8 : (!riscv.reg<>) -> !riscv.reg<>
  %load_double_known_offset = riscv.fld %load_double_ptr, 4 : (!riscv.reg<>) -> !riscv.freg<fa0>
  "test.op"(%load_double_known_offset) : (!riscv.freg<fa0>) -> ()

  %store_float_ptr = riscv.addi %i2, 8 : (!riscv.reg<>) -> !riscv.reg<>
  riscv.fsw %store_float_ptr, %f2, 4 : (!riscv.reg<>, !riscv.freg<>) -> ()

  %store_double_ptr = riscv.addi %i2, 8 : (!riscv.reg<>) -> !riscv.reg<>
  riscv.fsd %store_double_ptr, %f2, 4 : (!riscv.reg<>, !riscv.freg<>) -> ()

  %add_lhs_rhs = riscv.add %i1, %i1 : (!riscv.reg<a1>, !riscv.reg<a1>) -> !riscv.reg<a0>
  "test.op"(%add_lhs_rhs) : (!riscv.reg<a0>) -> ()

  %and_bitwise_zero_l0 = riscv.and %1, %0 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%and_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

  %and_bitwise_zero_r0 = riscv.and %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
  "test.op"(%and_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

  // scfgw immediates
  %scfgw = riscv_snitch.scfgw %i1, %1 : (!riscv.reg<a1>, !riscv.reg<>) -> !riscv.reg<zero>
  "test.op"(%scfgw) : (!riscv.reg<zero>) -> ()
}

// CHECK: builtin.module {
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<>)
// CHECK-NOT:    %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a1>) -> !riscv.reg<a2>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   "test.op"(%i0, %o1, %o2) : (!riscv.reg<a0>, !riscv.reg<a2>, !riscv.reg<>) -> ()

// CHECK-NEXT:   %i3 = riscv.li 100 : () -> !riscv.reg<>
// CHECK-NEXT:   %i4 = riscv.li 100 : () -> !riscv.reg<>
// CHECK-NEXT:   %i5 = riscv.li 100 : () -> !riscv.reg<j0>
// CHECK-NEXT:   "test.op"(%i3, %i4, %i5) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<j0>) -> ()

// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg<>)
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:   %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
// CHECK-NEXT:   %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:   "test.op"(%f0, %fo1, %fo2, %f0, %fo4, %fo5) : (!riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg<>, !riscv.freg<fa0>, !riscv.freg<fa2>, !riscv.freg<>) -> ()

// CHECK-NEXT:   %0 = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:   %1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:   %2 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %3 = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:   "test.op"(%0, %1, %2, %3) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

// CHECK-NEXT:   %multiply_immediates = riscv.li 6 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediate_r0 = riscv.li 0 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediate_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %multiply_immediate_l0 = riscv.li 0 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%multiply_immediate_l0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_lhs_immediate = riscv.addi %i2, 2 : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_lhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_rhs_immediate = riscv.addi %i2, 2 : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_rhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediates = riscv.li 5 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediates) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_vars = riscv.add %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediate_zero = riscv.mv %i2 : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediate_zero) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %add_immediate_constant = riscv.li 3 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_immediate_constant) : (!riscv.reg<a0>) -> ()

  // Unchanged
// CHECK-NEXT:   %sub_lhs_immediate = riscv.sub %2, %i2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_lhs_immediate) : (!riscv.reg<a0>) -> ()

  // Replace with addi
// CHECK-NEXT:   %sub_rhs_immediate = riscv.addi %i2, -2 : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_rhs_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %sub_immediates = riscv.li -1 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_immediates) : (!riscv.reg<a0>) -> ()

  // Unchanged
// CHECK-NEXT:   %sub_vars = riscv.sub %i0, %i1 : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_vars) : (!riscv.reg<a0>) -> ()

// Optimise out an arithmetic operation
// CHECK-NEXT:   %sub_add_immediate = riscv.li 2 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%sub_add_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %shift_left_immediate = riscv.li 32 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%shift_left_immediate) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %load_float_known_offset = riscv.flw %i2, 12 : (!riscv.reg<>) -> !riscv.freg<fa0>
// CHECK-NEXT:   "test.op"(%load_float_known_offset) : (!riscv.freg<fa0>) -> ()

// CHECK-NEXT:   %load_double_known_offset = riscv.fld %i2, 12 : (!riscv.reg<>) -> !riscv.freg<fa0>
// CHECK-NEXT:   "test.op"(%load_double_known_offset) : (!riscv.freg<fa0>) -> ()

// CHECK-NEXT:   riscv.fsw %i2, %f2, 12 : (!riscv.reg<>, !riscv.freg<>) -> ()

// CHECK-NEXT:   riscv.fsd %i2, %f2, 12 : (!riscv.reg<>, !riscv.freg<>) -> ()

// CHECK-NEXT:   %add_lhs_rhs = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %add_lhs_rhs_1 = riscv.mul %i1, %add_lhs_rhs : (!riscv.reg<a1>, !riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%add_lhs_rhs_1) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %and_bitwise_zero_l0 = riscv.li 0 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%and_bitwise_zero_l0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %and_bitwise_zero_r0 = riscv.li 0 : () -> !riscv.reg<a0>
// CHECK-NEXT:   "test.op"(%and_bitwise_zero_r0) : (!riscv.reg<a0>) -> ()

// CHECK-NEXT:   %scfgw = riscv_snitch.scfgwi %i1, 1 : (!riscv.reg<a1>) -> !riscv.reg<zero>
// CHECK-NEXT:   "test.op"(%scfgw) : (!riscv.reg<zero>) -> ()

// CHECK-NEXT: }

// -----

%ff0, %ff1 = "test.op"() : () -> (!riscv.freg<>, !riscv.freg<>)

// should fuse
%ffres0 = riscv.fmul.d %ff0, %ff1 {"fastmath" = #riscv.fastmath<fast>} : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
%ffres1 = riscv.fadd.d %ff0, %ffres0 {"fastmath" = #riscv.fastmath<fast>} : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

// should not fuse due to more than one uses
%ffres2 = riscv.fmul.d %ff0, %ff1: (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
%ffres3 = riscv.fadd.d %ff0, %ffres2 fastmath<fast> : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

// use multiplication result here to stop the fusion
"test.op"(%ffres2) : (!riscv.freg<>) -> ()

// keep results around to avoid dead code deletion
"test.op"(%ffres1, %ffres3) : (!riscv.freg<>, !riscv.freg<>) -> ()

// CHECK:        builtin.module {

// CHECK-NEXT:   %ff0, %ff1 = "test.op"() : () -> (!riscv.freg<>, !riscv.freg<>)

// CHECK-NEXT:   %ffres1 = riscv.fmadd.d %{{.*}}, %{{.*}}, %{{.*}} : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

// CHECK-NEXT:   %ffres2 = riscv.fmul.d %ff0, %ff1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:   %ffres3 = riscv.fadd.d %ff0, %ffres2 fastmath<fast> : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

// CHECK-NEXT:  "test.op"(%ffres2) : (!riscv.freg<>) -> ()

// CHECK-NEXT:  "test.op"(%ffres1, %ffres3) : (!riscv.freg<>, !riscv.freg<>) -> ()

// CHECK-NEXT:  }
