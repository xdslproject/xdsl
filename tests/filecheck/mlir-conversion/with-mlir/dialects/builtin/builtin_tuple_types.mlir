// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

"test.op"() {empty = tuple<>, many = tuple<i32, f32, tensor<i1>, i5>, single = tuple<f32>} : () -> ()

// CHECK: empty = tuple<>, many = tuple<i32, f32, tensor<i1>, i5>, single = tuple<f32>
