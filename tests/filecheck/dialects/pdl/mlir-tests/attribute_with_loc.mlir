// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @attribute_with_loc : benefit(1) {
    %0 = pdl.attribute
    %1 = pdl.operation {"attribute" = %0}
    pdl.rewrite %1 {
    }
  }
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @attribute_with_loc : benefit(1) {
// CHECK-NEXT:     %0 = pdl.attribute
// CHECK-NEXT:     %1 = pdl.operation {"attribute" = %0}
// CHECK-NEXT:     pdl.rewrite %1 {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
