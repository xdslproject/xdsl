// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%vector, %i0 = "test.op"() : () -> (vector<index>, index)

%0 = "vector.extractelement"(%vector, %i0) : (vector<index>, index) -> index
// CHECK: Expected position to be empty with 0-D vector.

// -----

%vector, %i0 = "test.op"() : () -> (vector<4x4xindex>, index)

%0 = "vector.extractelement"(%vector, %i0) : (vector<4x4xindex>, index) -> index
// CHECK: Operation does not verify: Unexpected >1 vector rank.

// -----

%vector, %i0= "test.op"() : () -> (vector<4xindex>, index)

%0 = "vector.extractelement"(%vector, %i0) : (vector<4xindex>, index) -> f64
// CHECK: Expected result type to match element type of vector operand.

// -----

%vector, %i0 = "test.op"() : () -> (vector<1xindex>, index)

%1 = "vector.extractelement"(%vector) : (vector<1xindex>) -> index
// CHECK: Expected position for 1-D vector.
