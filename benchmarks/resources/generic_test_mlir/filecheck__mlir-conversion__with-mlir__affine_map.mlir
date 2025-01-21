"builtin.module"() ({
  "f"() {map = affine_map<(d0, d1, d2)[s0, s1] -> (d0, d1, d2 + s0 + s1)>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> ()>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>} : () -> ()
  "f"() {map = affine_map<(d0, d1, d2) -> (d0 floordiv 2)>} : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "store_load"}> ({
    %0 = "test.op"() : () -> memref<2x3xf64>
    %1 = "test.op"() : () -> f64
    "affine.store"(%1, %0) <{map = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()
    %2 = "test.op"() : () -> index
    %3 = "affine.load"(%0, %2, %2) <{map = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
