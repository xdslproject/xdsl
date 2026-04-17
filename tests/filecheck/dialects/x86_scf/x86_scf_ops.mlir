// RUN: xdsl-opt %s | filecheck %s

%lb = x86.di.mov 0 : () -> !x86.reg64
%ub = x86.di.mov 100 : () -> !x86.reg64
%step = x86.di.mov 1 : () -> !x86.reg64
%acc = x86.di.mov 0 : () -> !x86.reg64<rbx>
%res_for = x86_scf.for %i : !x86.reg64 = %lb to %ub step %step iter_args(%acc_in = %acc) -> (!x86.reg64<rbx>) {
    %res = x86.ri.add %acc_in, 1 : (!x86.reg64<rbx>) -> !x86.reg64<rbx>
    x86_scf.yield %res : !x86.reg64<rbx>
}

%res_rof = x86_scf.rof %j : !x86.reg64 = %ub down to %lb step %step iter_args(%acc_in = %acc) -> (!x86.reg64<rbx>) {
    %res = x86.ri.add %acc_in, 1 : (!x86.reg64<rbx>) -> !x86.reg64<rbx>
    x86_scf.yield %res : !x86.reg64<rbx>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %lb = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:    %ub = x86.di.mov 100 : () -> !x86.reg64
// CHECK-NEXT:    %step = x86.di.mov 1 : () -> !x86.reg64
// CHECK-NEXT:    %acc = x86.di.mov 0 : () -> !x86.reg64<rbx>
// CHECK-NEXT:    %res_for = x86_scf.for %i : !x86.reg64  = %lb to %ub step %step iter_args(%acc_in = %acc) -> (!x86.reg64<rbx>) {
// CHECK-NEXT:      %res = x86.ri.add %acc_in, 1 : (!x86.reg64<rbx>) -> !x86.reg64<rbx>
// CHECK-NEXT:      x86_scf.yield %res : !x86.reg64<rbx>
// CHECK-NEXT:    }
// CHECK-NEXT:    %res_rof = x86_scf.rof %j : !x86.reg64  = %ub down  to %lb step %step iter_args(%acc_in_1 = %acc) -> (!x86.reg64<rbx>) {
// CHECK-NEXT:      %res_1 = x86.ri.add %acc_in_1, 1 : (!x86.reg64<rbx>) -> !x86.reg64<rbx>
// CHECK-NEXT:      x86_scf.yield %res_1 : !x86.reg64<rbx>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
