// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%vector, %i0 = "test.op"() : () -> (vector<index>, index)

%0 = "vector.insertelement"(%i0, %vector, %i0) : (index, vector<index>, index) -> vector<index>
// CHECK: Expected position to be empty with 0-D vector.

// -----

%vector, %i0 = "test.op"() : () -> (vector<1xindex>, index)

%1 = "vector.insertelement"(%i0, %vector) : (index, vector<1xindex>) -> vector<1xindex>
// CHECK: Expected position for 1-D vector.

// -----

%vector, %i0, %f0 = "test.op"() : () -> (vector<4xindex>, index, f64)

%0 = "vector.insertelement"(%f0, %vector, %i0) : (f64, vector<4xindex>, index) -> vector<4xindex>
// CHECK: Expected source operand type to match element type of dest operand.

// -----

%vector, %i0 = "test.op"() : () -> (vector<4xindex>, index)

%0 = "vector.insertelement"(%i0, %vector, %i0) : (index, vector<4xindex>, index) -> vector<3xindex>
// CHECK: Expected dest operand and result to have matching types.

// -----

%vector, %i0 = "test.op"() : () -> (vector<4x4xindex>, index)

%0 = "vector.insertelement"(%i0, %vector, %i0) : (index, vector<4x4xindex>, index) -> vector<4x4xindex>
// CHECK: Operation does not verify: Unexpected >1 vector rank.
