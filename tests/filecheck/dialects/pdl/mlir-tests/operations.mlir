// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @operations : benefit(1) {
    %0 = pdl.attribute
    %1 = pdl.type
    %2 = pdl.operation {"attr" = %0} -> (%1 : !pdl.type)
    %3 = pdl.result 0 of %2
    %4 = pdl.operand
    %5 = pdl.operation (%3, %4 : !pdl.value, !pdl.value)
    pdl.rewrite %5 {
    }
  }
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @operations : benefit(1) {
// CHECK-NEXT:     %0 = pdl.attribute
// CHECK-NEXT:     %1 = pdl.type
// CHECK-NEXT:     %2 = pdl.operation {"attr" = %0} -> (%1 : !pdl.type)
// CHECK-NEXT:     %3 = pdl.result 0 of %2
// CHECK-NEXT:     %4 = pdl.operand
// CHECK-NEXT:     %5 = pdl.operation (%3, %4 : !pdl.value, !pdl.value)
// CHECK-NEXT:     pdl.rewrite %5 {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
