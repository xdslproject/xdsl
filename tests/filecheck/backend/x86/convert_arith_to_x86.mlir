// RUN: xdsl-opt -p convert-arith-to-x86 --verify-diagnostics --split-input-file  %s | filecheck %s

%i0 = "test.op"(): () -> i32
%i1 = "test.op"(): () -> i32
%i2 = arith.addi %i0,%i1: i32

// CHECK:      builtin.module {
// CHECK-NEXT:   %i0 = "test.op"() : () -> i32
// CHECK-NEXT:   %i1 = "test.op"() : () -> i32
// CHECK-NEXT:   %i0_1 = builtin.unrealized_conversion_cast %i0 : i32 to !x86.reg
// CHECK-NEXT:   %i1_1 = builtin.unrealized_conversion_cast %i1 : i32 to !x86.reg
// CHECK-NEXT:   %i2 = x86.ds.mov %i1_1 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:   %i2_1 = x86.rs.add %i2, %i0_1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT:   %i2_2 = builtin.unrealized_conversion_cast %i2 : !x86.reg to i32
// CHECK-NEXT: }

// -----

// CHECK: Lowering of arith.addi not implemented for ShapedType
%i0 = "test.op"(): () -> tensor<2xi32>
%i1 = "test.op"(): () -> tensor<2xi32>
%i2 = arith.addi %i0,%i1: tensor<2xi32>

// -----

// CHECK: Not implemented for bitwidth larger than 64
%i0 = "test.op"(): () -> i128
%i1 = "test.op"(): () -> i128
%i2 = arith.addi %i0,%i1: i128
