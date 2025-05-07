// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s


%0, %1 = "test.op"() : () -> (vector<2xindex>, vector<3xindex>)

%2 = "vector.fma"(%0, %0, %1) : (vector<2xindex>, vector<2xindex>, vector<3xindex>) -> vector<2xindex>
// CHECK: Unexpected attribute index

// -----

%0, %1 = "test.op"() : () -> (vector<2xf32>, vector<3xf32>)

%2 = "vector.fma"(%0, %0, %1) : (vector<2xf32>, vector<2xf32>, vector<3xf32>) -> vector<2xf32>
// CHECK: attribute vector<2xf32> expected from variable 'T', but got vector<3xf32>

// -----

%0, %1 = "test.op"() : () -> (vector<2xf32>, vector<2xf64>)

%2 = "vector.fma"(%1, %1, %0) : (vector<2xf64>, vector<2xf64>, vector<2xf32>) -> vector<2xf64>
// CHECK: attribute vector<2xf64> expected from variable 'T', but got vector<2xf32>

// -----

%0, %1 = "test.op"() : () -> (vector<2xf32>, vector<3xf32>)

%2 = "vector.fma"(%1, %0, %0) : (vector<3xf32>, vector<2xf32>, vector<2xf32>) -> vector<2xf32>
// CHECK: attribute vector<3xf32> expected from variable 'T', but got vector<2xf32>

// -----

%0, %1 = "test.op"() : () -> (vector<2xf32>, vector<2xf64>)

%2 = "vector.fma"(%0, %1, %1) : (vector<2xf32>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
// CHECK: attribute vector<2xf32> expected from variable 'T', but got vector<2xf64>

// -----

%0, %1 = "test.op"() : () -> (vector<2xf32>, vector<3xf32>)

%2 = "vector.fma"(%0, %1, %0) : (vector<2xf32>, vector<3xf32>, vector<2xf32>) -> vector<2xf32>
// CHECK: attribute vector<2xf32> expected from variable 'T', but got vector<3xf32>

// -----

%0, %1 = "test.op"() : () -> (vector<2xf32>, vector<2xf64>)

%2 = "vector.fma"(%1, %0, %1) : (vector<2xf64>, vector<2xf32>, vector<2xf64>) -> vector<2xf64>
// CHECK: attribute vector<2xf64> expected from variable 'T', but got vector<2xf32>
