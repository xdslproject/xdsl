// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%N = "test.op"() : () -> index
"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<(i) -> (i)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
    "affine.yield"() : () -> ()
}) : (index) -> ()

// CHECK: Expected as many operands as results, lower bound args and upper bound args.

// -----

%N = "test.op"() : () -> index
"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<> : vector<0xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
    "affine.yield"() : () -> ()
}) : (index) -> ()

// CHECK: Expected a lower bound group for each lower bound

// -----

%N = "test.op"() : () -> index
"affine.parallel"(%N, %N) <{"lowerBoundsMap" = affine_map<()[s1] -> (0, 0, -s1)>, "lowerBoundsGroups" = dense<[1, 1, 2]> : vector<3xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index, %j : index):
    "affine.yield"() : () -> ()
}) : (index, index) -> ()

// CHECK: Expected a lower bound group for each lower bound


// -----

%N = "test.op"() : () -> index
"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<> : vector<0xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
    "affine.yield"() : () -> ()
}) : (index) -> ()

// CHECK: Expected an upper bound group for each upper bound
