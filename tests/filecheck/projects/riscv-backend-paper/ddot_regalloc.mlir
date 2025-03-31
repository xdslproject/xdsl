// RUN: xdsl-opt -p riscv-allocate-registers %s | filecheck %s

riscv.assembly_section ".text" {
    riscv_func.func public @ddot(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) attributes {p2align = 2 : i8} {
        %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg
        %Y_moved = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg
        %Z_moved = riscv.mv %Z : (!riscv.reg<a2>) -> !riscv.reg
        %init = riscv.fld %Z_moved, 0 : (!riscv.reg) -> !riscv.freg
        %lb = riscv.li 0 : !riscv.reg
        %ub = riscv.li 1024 : !riscv.reg
        %c8 = riscv.li 8 : !riscv.reg
        %res = riscv_scf.for %i : !riscv.reg = %lb to %ub step %c8 iter_args(%acc_in = %init) -> (!riscv.freg) {
            %x_ptr = riscv.add %X_moved, %i : (!riscv.reg, !riscv.reg) -> !riscv.reg
            %x = riscv.fld %x_ptr, 0 : (!riscv.reg) -> !riscv.freg
            %y_ptr = riscv.add %Y_moved, %i : (!riscv.reg, !riscv.reg) -> !riscv.reg
            %y = riscv.fld %y_ptr, 0 : (!riscv.reg) -> !riscv.freg
            %xy = riscv.fmul.d %x, %y : (!riscv.freg, !riscv.freg) -> !riscv.freg
            %acc_out = riscv.fadd.d %acc_in, %xy : (!riscv.freg, !riscv.freg) -> !riscv.freg
            riscv_scf.yield %acc_out : !riscv.freg
        }
        riscv.fsd %Z_moved, %res, 0 : (!riscv.reg, !riscv.freg) -> ()
        riscv_func.return
    }
}


// CHECK:       riscv.assembly_section ".text" {
// CHECK-NEXT:      riscv_func.func public @ddot(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>, %Z : !riscv.reg<a2>) attributes {p2align = 2 : i8} {
// CHECK-NEXT:          %X_moved = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg<t2>
// CHECK-NEXT:          %Y_moved = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg<t1>
// CHECK-NEXT:          %Z_moved = riscv.mv %Z : (!riscv.reg<a2>) -> !riscv.reg<t0>
// CHECK-NEXT:          %init = riscv.fld %Z_moved, 0 : (!riscv.reg<t0>) -> !riscv.freg<ft0>
// CHECK-NEXT:          %lb = riscv.li 0 : !riscv.reg<zero>
// CHECK-NEXT:          %ub = riscv.li 1024 : !riscv.reg<t4>
// CHECK-NEXT:          %c8 = riscv.li 8 : !riscv.reg<t5>
// CHECK-NEXT:          %res = riscv_scf.for %i : !riscv.reg<t3> = %lb to %ub step %c8 iter_args(%acc_in = %init) -> (!riscv.freg<ft0>) {
// CHECK-NEXT:              %x_ptr = riscv.add %X_moved, %i : (!riscv.reg<t2>, !riscv.reg<t3>) -> !riscv.reg<t6>
// CHECK-NEXT:              %x = riscv.fld %x_ptr, 0 : (!riscv.reg<t6>) -> !riscv.freg<ft1>
// CHECK-NEXT:              %y_ptr = riscv.add %Y_moved, %i : (!riscv.reg<t1>, !riscv.reg<t3>) -> !riscv.reg<t6>
// CHECK-NEXT:              %y = riscv.fld %y_ptr, 0 : (!riscv.reg<t6>) -> !riscv.freg<ft2>
// CHECK-NEXT:              %xy = riscv.fmul.d %x, %y : (!riscv.freg<ft1>, !riscv.freg<ft2>) -> !riscv.freg<ft1>
// CHECK-NEXT:              %acc_out = riscv.fadd.d %acc_in, %xy : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
// CHECK-NEXT:              riscv_scf.yield %acc_out : !riscv.freg<ft0>
// CHECK-NEXT:          }
// CHECK-NEXT:          riscv.fsd %Z_moved, %res, 0 : (!riscv.reg<t0>, !riscv.freg<ft0>) -> ()
// CHECK-NEXT:          riscv_func.return
// CHECK-NEXT:      }
// CHECK-NEXT:  }
