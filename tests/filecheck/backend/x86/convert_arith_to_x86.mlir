// RUN: xdsl-opt -p convert-arith-to-x86 --verify-diagnostics --split-input-file  %s | filecheck %s

%i0 = "test.op"(): () -> i32
%i1 = "test.op"(): () -> i32
%i2 = arith.addi %i0, %i1: i32

// CHECK:      builtin.module {
// CHECK-NEXT:   %i0 = "test.op"() : () -> i32
// CHECK-NEXT:   %i1 = "test.op"() : () -> i32
// CHECK-NEXT:   %i0_1 = builtin.unrealized_conversion_cast %i0 : i32 to !x86.reg
// CHECK-NEXT:   %i1_1 = builtin.unrealized_conversion_cast %i1 : i32 to !x86.reg
// CHECK-NEXT:   %i2 = x86.ds.mov %i1_1 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:   %i2_1 = x86.rs.add %i2, %i0_1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT:   %i2_2 = builtin.unrealized_conversion_cast %i2_1 : !x86.reg to i32
// CHECK-NEXT: }

// -----

// CHECK: Lowering of arith.addi not implemented for ShapedType
%i0 = "test.op"(): () -> tensor<2xi32>
%i1 = "test.op"(): () -> tensor<2xi32>
%i2 = arith.addi %i0, %i1: tensor<2xi32>

// -----

%i0 = "test.op"(): () -> i32
%i1 = "test.op"(): () -> i32
%i2 = arith.muli %i0, %i1: i32

// CHECK:      builtin.module {
// CHECK-NEXT:   %i0 = "test.op"() : () -> i32
// CHECK-NEXT:   %i1 = "test.op"() : () -> i32
// CHECK-NEXT:   %i0_1 = builtin.unrealized_conversion_cast %i0 : i32 to !x86.reg
// CHECK-NEXT:   %i1_1 = builtin.unrealized_conversion_cast %i1 : i32 to !x86.reg
// CHECK-NEXT:   %i2 = x86.ds.mov %i1_1 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:   %i2_1 = x86.rs.imul %i2, %i0_1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT:   %i2_2 = builtin.unrealized_conversion_cast %i2_1 : !x86.reg to i32
// CHECK-NEXT: }

// -----

// CHECK: Lowering of arith.muli not implemented for ShapedType
%i0 = "test.op"(): () -> tensor<2xi32>
%i1 = "test.op"(): () -> tensor<2xi32>
%i2 = arith.muli %i0,%i1: tensor<2xi32>

// -----

// CHECK: Not implemented for bitwidth larger than 64
%i0 = "test.op"(): () -> i128
%i1 = "test.op"(): () -> i128
%i2 = arith.addi %i0,%i1: i128

// -----

// CHECK: Lowering of arith.constant is only implemented for integers
%c = arith.constant 1.0: f32

// -----

%c = arith.constant 1: i32

// CHECK:      builtin.module {
// CHECK-NEXT:   %c = x86.di.mov 1 : () -> !x86.reg
// CHECK-NEXT:   %c_1 = builtin.unrealized_conversion_cast %c : !x86.reg to i32
// CHECK-NEXT: }

// -----

%c = arith.constant 1: index

// CHECK:      builtin.module {
// CHECK-NEXT:   %c = x86.di.mov 1 : () -> !x86.reg
// CHECK-NEXT:   %c_1 = builtin.unrealized_conversion_cast %c : !x86.reg to index
// CHECK-NEXT: }

// -----

// CHECK:         %f0, %f1 = "test.op"() : () -> (f32, f32)
%f0, %f1 = "test.op"(): () -> (f32, f32)
// CHECK-NEXT:    %f0_1 = builtin.unrealized_conversion_cast %f0 : f32 to !x86.reg
// CHECK-NEXT:    %f1_1 = builtin.unrealized_conversion_cast %f1 : f32 to !x86.reg
// CHECK-NEXT:    %addf = x86.ds.mov %f1_1 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:    %addf_1 = x86.rs.fadd %addf, %f0_1 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT:    %addf_2 = builtin.unrealized_conversion_cast %addf_1 : !x86.reg to f32
%addf = arith.addf %f0, %f1: f32
// CHECK-NEXT:    %f0_2 = builtin.unrealized_conversion_cast %f0 : f32 to !x86.reg
// CHECK-NEXT:    %f1_2 = builtin.unrealized_conversion_cast %f1 : f32 to !x86.reg
// CHECK-NEXT:    %mulf = x86.ds.mov %f1_2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:    %mulf_1 = x86.rs.fmul %mulf, %f0_2 : (!x86.reg, !x86.reg) -> !x86.reg
// CHECK-NEXT:    %mulf_2 = builtin.unrealized_conversion_cast %mulf_1 : !x86.reg to f32
%mulf = arith.mulf %f0, %f1: f32
