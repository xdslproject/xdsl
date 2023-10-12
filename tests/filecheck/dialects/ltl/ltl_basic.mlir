// RUN XDSL_ROUNDTRIP

%true = arith.constant true

// CHECK: unrealized_conversion_cast to !ltl.sequence
// CHECK: unrealized_conversion_cast to !ltl.property
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : i1, i1
// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : !ltl.property, !ltl.property
ltl.and %true, %true : i1, i1
ltl.and %s, %s : !ltl.sequence, !ltl.sequence
ltl.and %p, %p : !ltl.property, !ltl.property