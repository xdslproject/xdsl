// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %0 = pdl.types
    %1 = pdl.operation -> (%0 : !pdl.range<!pdl.type>)
    pdl.rewrite %1 {
      %2 = pdl.types : [i32, i64]
      %3 = pdl.operation "foo.op" -> (%0, %2 : !pdl.range<!pdl.type>, !pdl.range<!pdl.type>)
    }
  }
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:     %0 = pdl.types
// CHECK-NEXT:     %1 = pdl.operation -> (%0 : !pdl.range<!pdl.type>)
// CHECK-NEXT:     pdl.rewrite %1 {
// CHECK-NEXT:       %2 = pdl.types : [i32, i64]
// CHECK-NEXT:       %3 = pdl.operation "foo.op" -> (%0, %2 : !pdl.range<!pdl.type>, !pdl.range<!pdl.type>)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
