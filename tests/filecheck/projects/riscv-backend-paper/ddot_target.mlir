// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

riscv.assembly_section ".text" {
  riscv.directive ".globl" "ddot"
  riscv.directive ".p2align" "2"

  riscv_func.func @ddot(
    %X : !riscv.reg<a0>,
    %Y : !riscv.reg<a1>,
    %G : !riscv.reg<a2>
  ) {
    %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg<>
    %Y_moved = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg<>
    %G_moved = riscv.mv %G : (!riscv.reg<a2>) -> !riscv.reg<>

    "snitch_stream.streaming_region"(%X_moved, %Y_moved) <{
      "stride_patterns" = [#snitch_stream.stride_pattern<ub = [128], strides = [8]>],
      "operandSegmentSizes" = array<i32: 2, 0>
    }> ({
    ^bb0(%X_stream : !stream.readable<!riscv.freg<ft0>>, %Y_stream : !stream.readable<!riscv.freg<ft1>>):
        %init = riscv.fld %G_moved, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %c0 = riscv.li 0: () -> !riscv.reg<>
        %c1 = riscv.li 1: () -> !riscv.reg<>
        %c128 = riscv.li 128: () -> !riscv.reg<>
        %g = riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<ft0>
          %y = riscv_snitch.read from %Y_stream : !riscv.freg<ft1>
          %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<ft0>, !riscv.freg<ft1>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %G_moved, %g, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()
    }) : (!riscv.reg<>, !riscv.reg<>) -> ()

    riscv_func.return
  }
}

// CHECK:       .text
// CHECK-NEXT:  .globl ddot
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  ddot:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      li t4, 127
// CHECK-NEXT:      scfgwi t4, 95
// CHECK-NEXT:      scfgwi t3, 223
// CHECK-NEXT:      scfgwi t2, 768
// CHECK-NEXT:      scfgwi t1, 769
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      fld ft3, 0(t0)
// CHECK-NEXT:      li t1, 127
// CHECK-NEXT:      frep.o t1, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t0)
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret

