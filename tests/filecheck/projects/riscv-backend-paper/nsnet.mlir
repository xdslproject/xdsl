// RUN: xdsl-opt -p arith-add-fastmath,test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

func.func @main$async_dispatch_0_matmul_transpose_b_1x400x161_f64$xdsl_kernel1(%arg0: memref<1x161xf64>, %arg1: memref<5x161xf64, strided<[161, 1]>>, %arg2: memref<1x5xf64, strided<[40, 1]>>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<1x161xf64>, memref<5x161xf64, strided<[161, 1]>>) outs(%arg2 : memref<1x5xf64, strided<[40, 1]>>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %0 = arith.mulf %in, %in_0 : f64
    %1 = arith.addf %out, %0 : f64
    linalg.yield %1 : f64
  }
  return
}

// CHECK:  # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "a2", "zero"], "allocated_float": ["ft0", "ft1", "ft3", "ft4", "ft5", "ft6", "ft7"], "allocated_int": ["a0", "a1", "a2", "t0", "t1", "t2", "t3", "zero"]}
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl main$async_dispatch_0_matmul_transpose_b_1x400x161_f64$xdsl_kernel1
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  main$async_dispatch_0_matmul_transpose_b_1x400x161_f64$xdsl_kernel1:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t3, 160
// CHECK-NEXT:      scfgwi t3, 64
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 192
// CHECK-NEXT:      li t3, 4
// CHECK-NEXT:      scfgwi t3, 32
// CHECK-NEXT:      li t3, 4
// CHECK-NEXT:      scfgwi t3, 65
// CHECK-NEXT:      li t3, 160
// CHECK-NEXT:      scfgwi t3, 97
// CHECK-NEXT:      li t3, 1288
// CHECK-NEXT:      scfgwi t3, 193
// CHECK-NEXT:      li t3, -5144
// CHECK-NEXT:      scfgwi t3, 225
// CHECK-NEXT:      scfgwi zero, 33                              # dm 1 repeat
// CHECK-NEXT:      scfgwi t2, 768
// CHECK-NEXT:      scfgwi t1, 801
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      mv t1, t0
// CHECK-NEXT:      fld ft7, 0(t1)                               # load double from memref of shape (1, 5)
// CHECK-NEXT:      fld ft6, 8(t0)                               # load double from memref of shape (1, 5)
// CHECK-NEXT:      fld ft5, 16(t0)                              # load double from memref of shape (1, 5)
// CHECK-NEXT:      fld ft4, 24(t0)                              # load double from memref of shape (1, 5)
// CHECK-NEXT:      fld ft3, 32(t0)                              # load double from memref of shape (1, 5)
// CHECK-NEXT:      li t1, 160
// CHECK-NEXT:      frep.o t1, 5, 0, 0
// CHECK-NEXT:      fmadd.d ft7, ft0, ft1, ft7
// CHECK-NEXT:      fmadd.d ft6, ft0, ft1, ft6
// CHECK-NEXT:      fmadd.d ft5, ft0, ft1, ft5
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      mv t1, t0
// CHECK-NEXT:      fsd ft7, 0(t1)                               # store double value to memref of shape (1, 5)
// CHECK-NEXT:      fsd ft6, 8(t0)                               # store double value to memref of shape (1, 5)
// CHECK-NEXT:      fsd ft5, 16(t0)                              # store double value to memref of shape (1, 5)
// CHECK-NEXT:      fsd ft4, 24(t0)                              # store double value to memref of shape (1, 5)
// CHECK-NEXT:      fsd ft3, 32(t0)                              # store double value to memref of shape (1, 5)
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
