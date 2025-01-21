"builtin.module"() ({
  %0 = "test.op"() : () -> index
  "affine.parallel"(%0) <{lowerBoundsGroups = dense<1> : vector<1xi32>, lowerBoundsMap = affine_map<(d0) -> (d0)>, reductions = [], steps = [1], upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>}> ({
  ^bb0(%arg0: index):
    "affine.yield"() : () -> ()
  }) : (index) -> ()
}) : () -> ()
