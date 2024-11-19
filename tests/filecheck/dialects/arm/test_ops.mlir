// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP


// CHECK: %x1 = arm.get_register : !arm.reg<x1>
%x1 = arm.get_register : !arm.reg<x1>

// CHECK: %ds_mov = arm.ds.mov %x1 : (!arm.reg<x1>) -> !arm.reg<x2>
%ds_mov = arm.ds.mov %x1 : (!arm.reg<x1>) -> !arm.reg<x2>

// CHECK-GENERIC: %x1 = "arm.get_register"() : () -> !arm.reg<x1>
// CHECK-GENERIC: %ds_mov = "arm.ds.mov"(%x1) : (!arm.reg<x1>) -> !arm.reg<x2>
