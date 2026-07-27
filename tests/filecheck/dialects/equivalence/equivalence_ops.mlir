// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    %v0, %v1, %v2, %v3 = "test.op"() : () -> (index, index, index, f32)
%v0, %v1, %v2, %v3 = "test.op"() : () -> (index, index, index, f32)

// CHECK-NEXT:    %r0 = equivalence.class %v0 : index
// CHECK-NEXT:    %r1 = equivalence.class %v1, %v2 {hello = "world"} : index
%r0 = equivalence.class %v0 : index
%r1 = equivalence.class %v1, %v2 {"hello"="world"} : index

// CHECK-NEXT:    %r2 = equivalence.const_class %v3 (constant = -7.000000e+00 : f32) : f32
%r2 = equivalence.const_class %v3 (constant = -7.000000e+00 : f32) : f32


// CHECK-NEXT:    %egraph = equivalence.graph -> index {
// CHECK-NEXT:      %c = equivalence.class %r3 : index
// CHECK-NEXT:      %r3 = "test.op"(%r1) : (index) -> index
// CHECK-NEXT:      equivalence.yield %c : index
// CHECK-NEXT:    }

%egraph = equivalence.graph -> index {
    %c = equivalence.class %r3 : index
    %r3 = "test.op"(%r1) : (index) -> index
    equivalence.yield %c : index
}

// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %v0, %v1, %v2, %v3 = "test.op"() : () -> (index, index, index, f32)
// CHECK-GENERIC-NEXT:    %r0 = "equivalence.class"(%v0) : (index) -> index
// CHECK-GENERIC-NEXT:    %r1 = "equivalence.class"(%v1, %v2) {hello = "world"} : (index, index) -> index
// CHECK-GENERIC-NEXT:    %r2 = "equivalence.const_class"(%v3) <{value = -7.000000e+00 : f32}> : (f32) -> f32
// CHECK-GENERIC-NEXT:    %egraph = "equivalence.graph"() ({
// CHECK-GENERIC-NEXT:      %c = "equivalence.class"(%r3) : (index) -> index
// CHECK-GENERIC-NEXT:      %r3 = "test.op"(%r1) : (index) -> index
// CHECK-GENERIC-NEXT:      "equivalence.yield"(%c) : (index) -> ()
// CHECK-GENERIC-NEXT:    }) : () -> index
// CHECK-GENERIC-NEXT:  }) : () -> ()
