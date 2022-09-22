// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | FileCheck %s

"builtin.module"() ({
  "func.func"() ({
    ^bb0(%arg0: index):
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "index_type"} : () -> ()

  // CHECK: (index)


  "func.func"() ({
    ^bb0(%arg0: i32, %arg1: i64, %arg2: i1):
    "func.return"() : () -> ()
  }) {function_type = (i32, i64, i1) -> (), sym_name = "int_type"} : () -> ()

  // CHECK: (i32, i64, i1)


  "func.func"() ({
    ^bb0(%arg0: f16, %arg1: f32, %arg2: f64):
    "func.return"() : () -> ()
  }) {function_type = (f16, f32, f64) -> (), sym_name = "float_type"} : () -> ()

  // CHECK: (f16, f32, f64)


  "func.func"() ({}) {function_type = () -> (), value = 42.0 : f32, sym_name = "float_attr"} : () -> ()

  // CHECK: 42.0 : f32


  "func.func"() ({}) {function_type = () -> (), value = 42 : i32, sym_name = "int_attr"} : () -> ()

  // CHECK: 42 : i32


  "func.func"() ({}) {function_type = () -> (), value = "foo", sym_name = "string_attr"} : () -> ()

  // CHECK: "foo"


  "func.func"() ({}) {function_type = () -> (), value = [0, "foo"], sym_name = "list_attr"} : () -> ()

  // CHECK: [0 : i64, "foo"]


  "func.func"() ({
    ^bb0(%arg0: vector<4xf32>, %arg1: vector<f32>, %arg2: vector<1x12xi32>):
    "func.return"() : () -> ()
  }) {function_type = (vector<4xf32>, vector<f32>, vector<1x12xi32>) -> (), sym_name = "vector_type"} : () -> ()

  // CHECK: (vector<4xf32>, vector<f32>, vector<1x12xi32>)

  "func.func"() ({
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1x12xi32>, %arg3: tensor<*xf64>):
    "func.return"() : () -> ()
  }) {function_type = (tensor<4xf32>, tensor<f32>, tensor<1x12xi32>, tensor<*xf64>) -> (), sym_name = "tensor_type"} : () -> ()

  // CHECK: (tensor<4xf32>, tensor<f32>, tensor<1x12xi32>, tensor<*xf64>)

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<0> : tensor<1xi32>,
                      value2 = dense<[0.0, 1.0]> : tensor<2xf64>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: dense<[0]> : tensor<1xi32>
  // CHECK: dense<[0.0, 1.0]> : tensor<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = opaque<"test", "contents">
                      value2 = opaque<"test", "contents"> : tensor<2xf64>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: opaque<"test", "contents">
  // CHECK: opaque<"test", "contents"> : tensor<2xf64>

}) : () -> ()
