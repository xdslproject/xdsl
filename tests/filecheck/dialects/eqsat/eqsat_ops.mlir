// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    %v0, %v1 = "test.op"() : () -> (index, index)
%v0, %v1 = "test.op"() : () -> (index, index)

// CHECK-NEXT:    %r0 = eqsat.eclass %v0 : index
// CHECK-NEXT:    %r1 = eqsat.eclass %v0, %v1 {hello = "world"} : index
%r0 = eqsat.eclass %v0 : index
%r1 = eqsat.eclass %v0, %v1 {"hello"="world"} : index


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
// CHECK-GENERIC-NEXT:    %v0, %v1 = "test.op"() : () -> (index, index)
// CHECK-GENERIC-NEXT:    %r0 = "eqsat.eclass"(%v0) : (index) -> index
// CHECK-GENERIC-NEXT:    %r1 = "eqsat.eclass"(%v0, %v1) {hello = "world"} : (index, index) -> index
// CHECK-GENERIC-NEXT:    %egraph = "eqsat.egraph"() ({
// CHECK-GENERIC-NEXT:      %c = "eqsat.eclass"(%r3) : (index) -> index
// CHECK-GENERIC-NEXT:      %r3 = "test.op"(%r1) : (index) -> index
// CHECK-GENERIC-NEXT:      "eqsat.yield"(%c) : (index) -> ()
// CHECK-GENERIC-NEXT:    }) : () -> index
// CHECK-GENERIC-NEXT:  }) : () -> ()
