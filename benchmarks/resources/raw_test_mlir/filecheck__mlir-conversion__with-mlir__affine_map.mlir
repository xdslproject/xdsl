// RUN: xdsl-opt %s --allow-unregistered-dialect | mlir-opt --allow-unregistered-dialect --mlir-print-local-scope -mlir-print-op-generic | xdsl-opt --allow-unregistered-dialect

"builtin.module"() ({
  "f"() {map = affine_map<(d0, d1, d2)[s0, s1] -> (d0, d1, d2 + s0 + s1)>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> ()>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> (d0 floordiv 2)>} : () -> ()
  "func.func"() ({
    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    "affine.store"(%value, %memref) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()

    %zero = "test.op"() : () -> index
    %same_value = "affine.load"(%memref, %zero, %zero) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64

    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_load"} : () -> ()
}) : () -> ()
