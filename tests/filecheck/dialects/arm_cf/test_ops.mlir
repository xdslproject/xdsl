// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM

// CHECK: %x1 = arm.get_register : !arm.reg<x1>
%x1 = arm.get_register : !arm.reg<x1>

// CHECK: %x2 = arm.get_register : !arm.reg<x2>
%x2 = arm.get_register : !arm.reg<x2>

// CHECK: arm.label "testlabel" {comment = "this is a label"}
// CHECK-ASM: testlabel:                                       # this is a label
arm.label "testlabel" {comment = "this is a label"}

// CHECK-NEXT: arm_cf.beq %x1, %x2, ^testlabel
// CHECK-ASM: arm_cf.beq x1, x2, ^testlabel
arm_cf.beq %x1, %x2, ^testlabel : (!arm.reg<x1>, !arm.reg<x2>)

// CHECK-GENERIC: "arm.label"() <{label = "testlabel"}> {comment = "this is a label"} : () -> ()
