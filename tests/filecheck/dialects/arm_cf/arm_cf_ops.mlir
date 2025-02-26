// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: xdsl-opt -t arm-asm %s | filecheck %s --check-prefix=CHECK-ASM


// CHECK:       builtin.module {
// CHECK-NEXT:    arm_func.func @main() {
// CHECK-NEXT:      %x1 = arm.get_register : !arm.reg<x1>
// CHECK-NEXT:      %x2 = arm.get_register : !arm.reg<x2>
// CHECK-NEXT:      arm_cf.beq %x1, %x2, ^0 : (!arm.reg<x1>, !arm.reg<x2>)
// CHECK-NEXT:    ^0:
// CHECK-NEXT:      arm.label "testlabel" {comment = "this is a label"}
// CHECK-NEXT:      arm_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    "arm_func.func"() ({
// CHECK-GENERIC-NEXT:      %x1 = "arm.get_register"() : () -> !arm.reg<x1>
// CHECK-GENERIC-NEXT:      %x2 = "arm.get_register"() : () -> !arm.reg<x2>
// CHECK-GENERIC-NEXT:      "arm_cf.beq"(%x1, %x2) [^0] : (!arm.reg<x1>, !arm.reg<x2>) -> ()
// CHECK-GENERIC-NEXT:    ^0:
// CHECK-GENERIC-NEXT:      "arm.label"() <{label = "testlabel"}> {comment = "this is a label"} : () -> ()
// CHECK-GENERIC-NEXT:      "arm_func.return"() : () -> ()
// CHECK-GENERIC-NEXT:    }) {sym_name = "main", function_type = () -> ()} : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

// CHECK-ASM:           main:
// CHECK-ASM-NEXT:          beq x1, x2, testlabel
// CHECK-ASM-NEXT:      testlabel:                                       # this is a label
// CHECK-ASM-NEXT:          ret

arm_func.func @main() {
    %x1 = arm.get_register : !arm.reg<x1>
    %x2 = arm.get_register : !arm.reg<x2>
    arm_cf.beq %x1, %x2, ^bb0 : (!arm.reg<x1>, !arm.reg<x2>)
^bb0:
    arm.label "testlabel" {comment = "this is a label"}
    arm_func.return
}
