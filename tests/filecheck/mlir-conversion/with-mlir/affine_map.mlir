// RUN: xdsl-opt %s --allow-unregistered-dialect | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --allow-unregistered-dialect

"builtin.module"() ({
  "f"() {map = affine_map<(d0, d1, d2)[s0, s1] -> (d0, d1, d2 + s0 + s1)>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> ()>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> (d0 floordiv 2)>} : () -> ()
}) : () -> ()
