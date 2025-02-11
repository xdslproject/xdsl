// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM


// CHECK: %x1 = arm.get_register : !arm.reg<x1>
%x1 = arm.get_register : !arm.reg<x1>

// CHECK: %x2 = arm.get_register : !arm.reg<x2>
%x2 = arm.get_register : !arm.reg<x2>

// CHECK: %ds_mov = arm.ds.mov %x1 {comment = "move contents of s to d"} : (!arm.reg<x1>) -> !arm.reg<x2>
// CHECK-ASM: mov x2, x1 # move contents of s to d
%ds_mov = arm.ds.mov %x1 {"comment" = "move contents of s to d"} : (!arm.reg<x1>) -> !arm.reg<x2>

// CHECK: %dss_mul = arm.dss.mul %x1, %x2 {comment = "multiply s1 by s2"} : (!arm.reg<x1>, !arm.reg<x2>) -> !arm.reg<x3>
// CHECK-ASM: mul x3, x1, x2 # multiply s1 by s2
%dss_mul = arm.dss.mul %x1, %x2 {"comment" = "multiply s1 by s2"} : (!arm.reg<x1>, !arm.reg<x2>) -> !arm.reg<x3>

// CHECK-NEXT: arm.label "testlabel" {comment = "this is a label"}
// CHECK-ASM: testlabel:                                       # this is a label
arm.label "testlabel" {comment = "this is a label"}

// CHECK-NEXT: arm.cmp %x2, %x1 {comment = "compare values in x2 and x1"}
// CHECK-ASM: cmp x2, x1 # compare values in x2 and x1
arm.cmp %x2, %x1 {comment = "compare values in x2 and x1"} : (!arm.reg<x2>, !arm.reg<x1>)

// CHECK-GENERIC: %x1 = "arm.get_register"() : () -> !arm.reg<x1>
// CHECK-GENERIC: %ds_mov = "arm.ds.mov"(%x1) {comment = "move contents of s to d"} : (!arm.reg<x1>) -> !arm.reg<x2>
// CHECK-GENERIC: %dss_mul = "arm.dss.mul"(%x1, %x2) {comment = "multiply s1 by s2"} : (!arm.reg<x1>, !arm.reg<x2>) -> !arm.reg<x3>
// CHECK-GENERIC: "arm.label"() <{label = "testlabel"}> {comment = "this is a label"} : () -> ()
// CHECK-GENERIC: "arm.cmp"(%x2, %x1) {comment = "compare values in x2 and x1"} : (!arm.reg<x2>, !arm.reg<x1>) -> ()
