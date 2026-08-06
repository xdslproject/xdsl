// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%N = "test.op"() : () -> index
"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<(i) -> (i)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^bb1(%i: index):
    "affine.yield"() : () -> ()
}) : (index) -> ()

// CHECK: Expected as many operands as results, lower bound args and upper bound args.

// -----

%N = "test.op"() : () -> index
"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<> : vector<0xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^bb1(%i: index):
    "affine.yield"() : () -> ()
}) : (index) -> ()

// CHECK: Expected a lower bound group for each lower bound

// -----

%N = "test.op"() : () -> index
"affine.parallel"(%N, %N) <{"lowerBoundsMap" = affine_map<()[s1] -> (0, 0, -s1)>, "lowerBoundsGroups" = dense<[1, 1, 2]> : vector<3xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^bb1(%i: index, %j: index):
    "affine.yield"() : () -> ()
}) : (index, index) -> ()

// CHECK: Expected a lower bound group for each lower bound


// -----

%N = "test.op"() : () -> index
"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<> : vector<0xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^bb1(%i: index):
    "affine.yield"() : () -> ()
}) : (index) -> ()

// CHECK: Expected an upper bound group for each upper bound

// -----

%not_memref = "test.op"() : () -> tensor<2x3xf64>
%value = "test.op"() : () -> f64
affine.store %value, %not_memref[0, 0] : tensor<2x3xf64>

// CHECK: Expected memref type

// -----

%memref = "test.op"() : () -> memref<2x3xf64>
%vector = affine.vector_load %memref[0, 0] : memref<2x3xf64>, tensor<f64>

// CHECK: Expected affine.vector_load to return a vector, but found: tensor<f64>

// -----

%d0 = "test.op"() : () -> index
%d1 = "test.op"() : () -> index
affine.for %i = affine_map<(d0, d1) -> (d0, d1)>(%d0, %d1) to 10 {
}

// CHECK: loop bound affine map with multiple results requires 'max' prefix

// -----

%ub = "test.op"() : () -> index
affine.for %i = 0 to affine_map<(d0, d1) -> (d0, d1)>(%ub) {
}

// CHECK: dim operand count and affine map dim count must match

// -----

affine.for %i = 0 to 10 step -2 {
}

// CHECK: expected step to be representable as a positive signed integer
