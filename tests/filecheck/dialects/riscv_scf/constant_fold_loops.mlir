// RUN: xdsl-opt -p convert-func-to-riscv-func,convert-memref-to-riscv,convert-arith-to-riscv,convert-scf-to-riscv-scf,dce,reconcile-unrealized-casts,canonicalize %s | filecheck %s

builtin.module {
  func.func public @ssum(%arg0: f32, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %0 = memref.load %arg1[%arg4] : memref<64xf32>
      %1 = memref.load %arg2[%arg4] : memref<64xf32>
      %2 = arith.mulf %0, %arg0 : f32
      %3 = arith.addf %2, %1 : f32
      memref.store %3, %arg3[%arg4] : memref<64xf32>
      scf.yield
    }
    func.return
  }
}


// CHECK:       builtin.module {
// CHECK-NEXT:    riscv.assembly_section ".text" {
// CHECK-NEXT:      riscv.directive ".globl" "ssum" : () -> ()
// CHECK-NEXT:      riscv.directive ".p2align" "2" : () -> ()
// CHECK-NEXT:      riscv_func.func @ssum(%arg0 : !riscv.freg<fa0>, %arg1 : !riscv.reg<a1>, %arg2 : !riscv.reg<a2>, %arg3 : !riscv.reg<a3>) {
// CHECK-NEXT:        %{{.*}} = riscv.fmv.s %arg0 : (!riscv.freg<fa0>) -> !riscv.freg<>
// CHECK-NEXT:        %{{.*}} = riscv.mv %arg1 : (!riscv.reg<a1>) -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mv %arg2 : (!riscv.reg<a2>) -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mv %arg3 : (!riscv.reg<a3>) -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 256 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:        riscv_scf.for %arg4 : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          %{{.*}} = riscv.add %{{.*}}, %arg4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = riscv.flw %{{.*}}, 0 {"comment" = "load value from memref of shape (64,)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.add %{{.*}}, %arg4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = riscv.flw %{{.*}}, 0 {"comment" = "load value from memref of shape (64,)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.fmul.s %{{.*}}, %0 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.fadd.s %{{.*}}, %{{.*}} : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.add %{{.*}}, %arg4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          riscv.fsw %{{.*}}, %{{.*}}, 0 {"comment" = "store float value to memref of shape (64,)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:          riscv_scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:        riscv_func.return
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
