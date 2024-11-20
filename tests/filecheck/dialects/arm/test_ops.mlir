// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM


// CHECK: %x1 = arm.get_register : !arm.reg<x1>
%x1 = arm.get_register : !arm.reg<x1>

// CHECK: %ds_mov = arm.ds.mov %x1 {"comment" = "move contents of d to s"} : (!arm.reg<x1>) -> !arm.reg<x2>
// CHECK-ASM: mov x2, x1 # move contents of d to s
%ds_mov = arm.ds.mov %x1 {"comment" = "move contents of d to s"} : (!arm.reg<x1>) -> !arm.reg<x2>

// CHECK-GENERIC: %x1 = "arm.get_register"() : () -> !arm.reg<x1>
// CHECK-GENERIC: %ds_mov = "arm.ds.mov"(%x1) {"comment" = "move contents of d to s"} : (!arm.reg<x1>) -> !arm.reg<x2>
