// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

func.func public @exp_f64(
  %X: memref<8xf64>,
  %Y: memref<8xf64>
) {
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%X : memref<8xf64>) outs(%Y : memref<8xf64>) {
  ^bb0(%in: f64, %out: f64):
    %0 = math.exp %in {terms = 4 : i64} : f64
    linalg.yield %0 : f64
  }
  func.return
}

// CHECK:  # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "sp", "zero"], "allocated_float": ["ft0", "ft1", "ft3", "ft4", "ft5", "ft6"], "allocated_int": ["a0", "a1", "sp", "t0", "t1", "t2", "t3", "zero"]}
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl exp_f64
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  exp_f64:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      li t0, 7
// CHECK-NEXT:      scfgwi t0, 95                                # dm 31 dim 0 bound
// CHECK-NEXT:      li t0, 8
// CHECK-NEXT:      scfgwi t0, 223                               # dm 31 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
// CHECK-NEXT:      scfgwi t2, 768                               # dm 0 dim 0 source
// CHECK-NEXT:      scfgwi t1, 897                               # dm 1 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      mv t0, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_0_for:
// CHECK-NEXT:      fmv.d ft4, ft0
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      fcvt.d.w ft3, t3
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      fcvt.d.w ft5, t3
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      fcvt.d.w ft6, t3
// CHECK-NEXT:      fmul.d ft6, ft4, ft6
// CHECK-NEXT:      fmul.d ft6, ft6, ft5
// CHECK-NEXT:      fadd.d ft3, ft3, ft6
// CHECK-NEXT:      li t3, 1071644672
// CHECK-NEXT:      sw t3, -4(sp)
// CHECK-NEXT:      sw zero, -8(sp)
// CHECK-NEXT:      fld ft5, -8(sp)
// CHECK-NEXT:      fmul.d ft5, ft4, ft5
// CHECK-NEXT:      fmul.d ft5, ft5, ft6
// CHECK-NEXT:      fadd.d ft3, ft3, ft5
// CHECK-NEXT:      li t3, 1070945621
// CHECK-NEXT:      sw t3, -4(sp)
// CHECK-NEXT:      li t3, 1431655765
// CHECK-NEXT:      sw t3, -8(sp)
// CHECK-NEXT:      fld ft6, -8(sp)
// CHECK-NEXT:      fmul.d ft4, ft4, ft6
// CHECK-NEXT:      fmul.d ft4, ft4, ft5
// CHECK-NEXT:      fadd.d ft1, ft3, ft4
// CHECK-NEXT:      addi t0, t0, 1
// CHECK-NEXT:      blt t0, t1, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      ret
