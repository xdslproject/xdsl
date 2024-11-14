// RUN: xdsl-opt -t arm-asm %s | filecheck %s

%0 = arm.get_register : () -> !arm.reg<x0>
%1 = arm.get_register : () -> !arm.reg<x1>

%rr_mov = arm.rr.mov %0, %1 : (!arm.reg<x0>, !arm.reg<x1>) -> !arm.reg<x0>
// CHECK: mov x0, x1
