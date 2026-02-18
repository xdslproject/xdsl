// RUN: xdsl-opt -p convert-riscv-scf-for-to-frep %s | filecheck %s

%i0 = rv32.get_register : !riscv.reg
%i1 = rv32.get_register : !riscv.reg
%i2 = rv32.get_register : !riscv.reg

%c1 = rv32.li 1 : !riscv.reg
%c2 = rv32.li 2 : !riscv.reg

%readable = riscv_snitch.get_stream : !snitch.readable<!riscv.freg<ft0>>
%writable = riscv_snitch.get_stream : !snitch.writable<!riscv.freg<ft1>>

%f0 = riscv.get_float_register : !riscv.freg<ft2>
%f1 = riscv.get_float_register : !riscv.freg<ft3>
%f2 = riscv.get_float_register : !riscv.freg<ft4>

// Success

riscv_scf.for %index0 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
    %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    %f5 = riscv.fadd.d %f4, %f4 : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
    riscv_snitch.write %f5 to %writable : !riscv.freg<ft1>
}

%res = riscv_scf.for %index1 : !riscv.reg<a4> = %i0 to %i1 step %c1 iter_args(%f3 = %f0) -> (!riscv.freg<ft2>) {
    %f4 = riscv.fadd.d %f3, %f3 : (!riscv.freg<ft2>, !riscv.freg<ft2>) -> !riscv.freg<ft2>
    riscv_scf.yield %f4 : !riscv.freg<ft2>
}

// CHECK:         riscv_snitch.frep_outer %1 {
// CHECK-NEXT:      %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
// CHECK-NEXT:      %f5 = riscv.fadd.d %f4, %f4 : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
// CHECK-NEXT:      riscv_snitch.write %f5 to %writable : !riscv.freg<ft1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %res = riscv.sub %i1, %i0 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %res_1 = riscv.addi %res, -1 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %res_2 = riscv_snitch.frep_outer %res_1 iter_args(%f3 = %f0) -> (!riscv.freg<ft2>) {
// CHECK-NEXT:      %{{.*}} = riscv.fadd.d %f3, %f3 : (!riscv.freg<ft2>, !riscv.freg<ft2>) -> !riscv.freg<ft2>
// CHECK-NEXT:      riscv_snitch.frep_yield %{{.*}} : !riscv.freg<ft2>
// CHECK-NEXT:    }

// Failure


// 1. Induction variable is used

riscv_scf.for %index2 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
    %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    %f5 = riscv.fcvt.s.w %index2 : (!riscv.reg<a4>) -> !riscv.freg<ft1>
    riscv_snitch.write %f5 to %writable : !riscv.freg<ft1>
}

// CHECK-NEXT:    riscv_scf.for %index2 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
// CHECK-NEXT:      %{{.*}} = riscv_snitch.read from %readable : !riscv.freg<ft0>
// CHECK-NEXT:      %{{.*}} = riscv.fcvt.s.w %index2 : (!riscv.reg<a4>) -> !riscv.freg<ft1>
// CHECK-NEXT:      riscv_snitch.write %{{.*}} to %writable : !riscv.freg<ft1>
// CHECK-NEXT:    }

// 2. Step is 1

riscv_scf.for %index3 : !riscv.reg<a4> = %i0 to %i1 step %c2 {
    %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    %f5 = riscv.fadd.d %f4, %f4 : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
    riscv_snitch.write %f5 to %writable : !riscv.freg<ft1>
}

// CHECK-NEXT:    riscv_scf.for %index3 : !riscv.reg<a4> = %i0 to %i1 step %c2 {
// CHECK-NEXT:      %{{.*}} = riscv_snitch.read from %readable : !riscv.freg<ft0>
// CHECK-NEXT:      %{{.*}} = riscv.fadd.d %{{.*}}, %{{.*}} : (!riscv.freg<ft0>, !riscv.freg<ft0>) -> !riscv.freg<ft1>
// CHECK-NEXT:      riscv_snitch.write %{{.*}} to %writable : !riscv.freg<ft1>
// CHECK-NEXT:    }


// 3. All operations in the loop all operate on float registers

riscv_scf.for %index4 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
    %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    %f5 = riscv.fcvt.s.w %i2 : (!riscv.reg) -> !riscv.freg<ft1>
    riscv_snitch.write %f5 to %writable : !riscv.freg<ft1>
}

// CHECK-NEXT:    riscv_scf.for %index4 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
// CHECK-NEXT:      %{{.*}} = riscv_snitch.read from %readable : !riscv.freg<ft0>
// CHECK-NEXT:      %{{.*}} = riscv.fcvt.s.w %i2 : (!riscv.reg) -> !riscv.freg<ft1>
// CHECK-NEXT:      riscv_snitch.write %{{.*}} to %writable : !riscv.freg<ft1>
// CHECK-NEXT:    }



// 4. All operations are pure or one of
//      a) riscv_snitch.read
//      b) riscv_snitch.write
//      c) builtin.unrealized_conversion_cast

riscv_scf.for %index5 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
    %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
    "test.op"(%f4) : (!riscv.freg<ft0>) -> ()
}

// CHECK-NEXT:    riscv_scf.for %index5 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
// CHECK-NEXT:      %{{.*}} = riscv_snitch.read from %readable : !riscv.freg<ft0>
// CHECK-NEXT:      "test.op"(%{{.*}}) : (!riscv.freg<ft0>) -> ()
// CHECK-NEXT:    }
