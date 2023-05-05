// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @rewrite_multi_root_forced : benefit(2) {
    %0 = pdl.operand
    %1 = pdl.operand
    %2 = pdl.type
    %3 = pdl.operation (%0 : !pdl.value) -> (%2 : !pdl.type)
    %4 = pdl.result 0 of %3
    %5 = pdl.operation (%4 : !pdl.value)
    %6 = pdl.operation (%1 : !pdl.value) -> (%2 : !pdl.type)
    %7 = pdl.result 0 of %6
    %8 = pdl.operation (%4, %7 : !pdl.value, !pdl.value)
    pdl.rewrite %5 {
    }
  }
}) : () -> ()


// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @rewrite_multi_root_forced : benefit(2) {
// CHECK-NEXT:     %0 = pdl.operand
// CHECK-NEXT:     %1 = pdl.operand
// CHECK-NEXT:     %2 = pdl.type
// CHECK-NEXT:     %3 = pdl.operation (%0 : !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:     %4 = pdl.result 0 of %3
// CHECK-NEXT:     %5 = pdl.operation (%4 : !pdl.value)
// CHECK-NEXT:     %6 = pdl.operation (%1 : !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:     %7 = pdl.result 0 of %6
// CHECK-NEXT:     %8 = pdl.operation (%4, %7 : !pdl.value, !pdl.value)
// CHECK-NEXT:     pdl.rewrite %5 {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
