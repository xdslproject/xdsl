// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    %v0, %v1 = "test.op"() : () -> (index, index)
%v0, %v1 = "test.op"() : () -> (index, index)

// CHECK-NEXT:    %r0 = eqsat.eclass %v0 : index
// CHECK-NEXT:    %r1 = eqsat.eclass %v0, %v1 {hello = "world"} : index
%r0 = eqsat.eclass %v0 : index
%r1 = eqsat.eclass %v0, %v1 {"hello"="world"} : index

// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %v0, %v1 = "test.op"() : () -> (index, index)
// CHECK-GENERIC-NEXT:    %r0 = "eqsat.eclass"(%v0) : (index) -> index
// CHECK-GENERIC-NEXT:    %r1 = "eqsat.eclass"(%v0, %v1) {hello = "world"} : (index, index) -> index
// CHECK-GENERIC-NEXT:  }) : () -> ()
