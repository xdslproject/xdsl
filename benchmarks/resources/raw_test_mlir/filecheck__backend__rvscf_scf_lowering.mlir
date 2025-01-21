// RUN: xdsl-opt -p convert-scf-to-riscv-scf %s | filecheck %s

builtin.module {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %i_in = arith.constant 42 : index
  %f32_in = arith.constant 21.0 : f32
  %f64_in = arith.constant 42.0 : f64

  %i_out, %f32_out, %f64_out = scf.for %idx = %c0 to %c10 step %c1 iter_args(%i_acc = %i_in, %f32_acc = %f32_in, %f64_acc = %f64_in) -> (index, f32, f64) {
    %res = arith.addi %idx, %i_acc : index
    scf.yield %res, %f32_acc, %f64_acc : index, f32, f64
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c10 = arith.constant 10 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %i_in = arith.constant 42 : index
// CHECK-NEXT:   %f32_in = arith.constant 2.100000e+01 : f32
// CHECK-NEXT:   %f64_in = arith.constant 4.200000e+01 : f64
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f32 to !riscv.freg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f64 to !riscv.freg
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = riscv_scf.for %idx : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!riscv.reg, !riscv.freg, !riscv.freg) {
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f64
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to index
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to index
// CHECK-NEXT:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : index to !riscv.reg
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f32 to !riscv.freg
// CHECK-NEXT:     %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f64 to !riscv.freg
// CHECK-NEXT:     riscv_scf.yield %{{.*}}, %{{.*}}, %{{.*}} : !riscv.reg, !riscv.freg, !riscv.freg
// CHECK-NEXT:   }
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to index
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f64
// CHECK-NEXT: }
