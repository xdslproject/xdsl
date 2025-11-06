// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    %v0, %v1, %v2, %v3 = "test.op"() : () -> (index, index, index, f32)
%v0, %v1, %v2, %v3 = "test.op"() : () -> (index, index, index, f32)

// CHECK-NEXT:    %r0 = eqsat.eclass %v0 : index
// CHECK-NEXT:    %r1 = eqsat.eclass %v1, %v2 {hello = "world"} : index
%r0 = eqsat.eclass %v0 : index
%r1 = eqsat.eclass %v1, %v2 {"hello"="world"} : index

// CHECK-NEXT:    %r2 = eqsat.const_eclass %v3 (constant = -7.000000e+00 : f32) : f32
%r2 = eqsat.const_eclass %v3 (constant = -7.000000e+00 : f32) : f32


// CHECK-NEXT:    %egraph = eqsat.egraph -> index {
// CHECK-NEXT:      %c = eqsat.eclass %r3 : index
// CHECK-NEXT:      %r3 = "test.op"(%r1) : (index) -> index
// CHECK-NEXT:      eqsat.yield %c : index
// CHECK-NEXT:    }

%egraph = eqsat.egraph -> index {
    %c = eqsat.eclass %r3 : index
    %r3 = "test.op"(%r1) : (index) -> index
    eqsat.yield %c : index
}

// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %v0, %v1, %v2, %v3 = "test.op"() : () -> (index, index, index, f32)
// CHECK-GENERIC-NEXT:    %r0 = "eqsat.eclass"(%v0) : (index) -> index
// CHECK-GENERIC-NEXT:    %r1 = "eqsat.eclass"(%v1, %v2) {hello = "world"} : (index, index) -> index
// CHECK-GENERIC-NEXT:    %r2 = "eqsat.const_eclass"(%v3) <{value = -7.000000e+00 : f32}> : (f32) -> f32
// CHECK-GENERIC-NEXT:    %egraph = "eqsat.egraph"() ({
// CHECK-GENERIC-NEXT:      %c = "eqsat.eclass"(%r3) : (index) -> index
// CHECK-GENERIC-NEXT:      %r3 = "test.op"(%r1) : (index) -> index
// CHECK-GENERIC-NEXT:      "eqsat.yield"(%c) : (index) -> ()
// CHECK-GENERIC-NEXT:    }) : () -> index
// CHECK-GENERIC-NEXT:  }) : () -> ()
