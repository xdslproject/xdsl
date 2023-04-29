// RUN: xdsl-opt %s -t mlir --allow-unregistered-dialect | xdsl-opt -f mlir -t mlir --allow-unregistered-dialect  | filecheck %s

"builtin.module"() ({

  "test"() {affine_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4 * 3, d2 * 2 + d5 * 4, d3)>} : () -> ()
  "test"() {affine_map = affine_set<(d0) : (-d0 + 20 >= 0)>} : () -> ()

  // CHECK:     "test"() {"affine_map" = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4 * 3, d2 * 2 + d5 * 4, d3)>} : () -> ()
  // CHECK-NEXT:"test"() {"affine_map" = affine_set<(d0) : (-d0 + 20 >= 0)>} : () -> ()

}) : () -> ()
