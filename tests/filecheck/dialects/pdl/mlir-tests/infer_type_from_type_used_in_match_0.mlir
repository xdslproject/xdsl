// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %0 = pdl.type : i32
    %1 = pdl.type
    %2 = pdl.operation -> (%0, %1 : !pdl.type, !pdl.type)
    pdl.rewrite %2 {
      %3 = pdl.operation "foo.op" -> (%0, %1 : !pdl.type, !pdl.type)
    }
  }
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:     %0 = pdl.type : i32
// CHECK-NEXT:     %1 = pdl.type
// CHECK-NEXT:     %2 = pdl.operation -> (%0, %1 : !pdl.type, !pdl.type)
// CHECK-NEXT:     pdl.rewrite %2 {
// CHECK-NEXT:       %3 = pdl.operation "foo.op" -> (%0, %1 : !pdl.type, !pdl.type)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
