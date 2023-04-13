// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s

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
    ^bb0(%arg0: si32, %arg1: si64, %arg2: si1):
    "func.return"() : () -> ()
  }) {function_type = (si32, si64, si1) -> (), sym_name = "signed_int_type"} : () -> ()

  // CHECK: (si32, si64, si1)

  "func.func"() ({
    ^bb0(%arg0: ui32, %arg1: ui64, %arg2: ui1):
    "func.return"() : () -> ()
  }) {function_type = (ui32, ui64, ui1) -> (), sym_name = "unsigned_int_type"} : () -> ()

  // CHECK: (ui32, ui64, ui1)

  "func.func"() ({
    ^bb0(%arg0: bf16, %arg1: f16, %arg2: f32, %arg3: f64, %arg3: f80, %arg3: f128):
    "func.return"() : () -> ()
  }) {function_type = (bf16, f16, f32, f64, f80, f128) -> (), sym_name = "float_type"} : () -> ()

  // CHECK: (bf16, f16, f32, f64, f80, f128)


  "func.func"() ({}) {function_type = () -> (), value = 42.0 : f32, sym_name = "float_attr"} : () -> ()

  // CHECK: 42.0 : f32

  "func.func"() ({}) {function_type = () -> (), value = true, sym_name = "true_attr"} : () -> ()

  // CHECK: "value" = true

  "func.func"() ({}) {function_type = () -> (), value = 1 : i1, sym_name = "true_explicit_attr"} : () -> ()

  // CHECK: "value" = true

  "func.func"() ({}) {function_type = () -> (), value = false, sym_name = "false_attr"} : () -> ()

  // CHECK: "value" = false

  "func.func"() ({}) {function_type = () -> (), value = 0 : i1, sym_name = "false_explicit_attr"} : () -> ()

  // CHECK: "value" = false


  "func.func"() ({}) {function_type = () -> (), value = 42 : i32, sym_name = "int_attr"} : () -> ()

  // CHECK: 42 : i32

  "func.func"() ({}) {function_type = () -> (), value = 54 : index, sym_name = "index_attr"} : () -> ()

  // CHECK: 54 : index

  "func.func"() ({}) {function_type = () -> (), value = 54 : f32, sym_name = "index_attr"} : () -> ()

  // CHECK: 54.0 : f32

  "func.func"() ({}) {function_type = () -> (), value = 0x132 : i32, sym_name = "index_attr"} : () -> ()

  // CHECK: 306 : i32

  "func.func"() ({}) {function_type = () -> (), value = 0x132 : f32, sym_name = "index_attr"} : () -> ()

  // CHECK: 306.0 : f32


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
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1x12xi32>, %arg3: tensor<*xf64>, %arg4: tensor<0xi32>):
    "func.return"() : () -> ()
  }) {function_type = (tensor<4xf32>, tensor<f32>, tensor<1x12xi32>, tensor<*xf64>, tensor<0xi32>) -> (), sym_name = "tensor_type"} : () -> ()

  // CHECK: (tensor<4xf32>, tensor<f32>, tensor<1x12xi32>, tensor<*xf64>, tensor<0xi32>)

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<[[2, 3]]> : tensor<1x2xi32>,
                      value2 = dense<[0.0, 1.0]> : tensor<2xf64>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "value1" = dense<[[2, 3]]> : tensor<1x2xi32>, "value2" = dense<[0.0, 1.0]> : tensor<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<[0]> : vector<1xi32>,
                      value2 = dense<[0.0, 1.0]> : vector<2xf64>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "value1" = dense<0> : vector<1xi32>, "value2" = dense<[0.0, 1.0]> : vector<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<> : tensor<1x23x0x4xi32>,
                      value2 = dense<[[0.0], [1.0]]> : tensor<2x1xf64>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "value1" = dense<> : tensor<1x23x0x4xi32>, "value2" = dense<[[0.0], [1.0]]> : tensor<2x1xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<12> : tensor<2x3xi32>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "value1" = dense<12> : tensor<2x3xi32>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<[true, false]> : tensor<2xi1>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "value1" = dense<[1, 0]> : tensor<2xi1>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = opaque<"test", "contents">,
                      value2 = opaque<"test", "contents"> : tensor<2xf64>,
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "value1" = opaque<"test", "contents">, "value2" = opaque<"test", "contents"> : tensor<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value = {"one"=1, "two"=2, "three"="three"},
                      sym_name = "dense_attr"} : () -> ()

  // CHECK: "one"=1 : i64, "two"=2 : i64, "three"="three"

  "func.func"() ({}) {function_type = () -> (),
                      symbol = @some_symbol,
                      sym_name = "symbol_attr"} : () -> ()

  // CHECK: "symbol" = @some_symbol

  "func.func"() ({}) {function_type = () -> (),
                      value1 = tensor<?xi32>,
                      sym_name = "non_static_shaped_tensor"} : () -> ()

  // CHECK: tensor<?xi32>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = tensor<?xi32, "encoding">,
                      sym_name = "tensor_with_encoding"} : () -> ()

  // CHECK: tensor<?xi32, "encoding">

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<2xf32>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<2xf32>

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<2x?xf32>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<2x?xf32>

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<2xf32, strided<[]>>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<2xf32, strided<[]>>

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<2xf32, strided<[]>, 2>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<2xf32, strided<[]>, 2 : i64>

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<2xf32, 2>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<2xf32, 2 : i64>

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<*xf32>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<*xf32>

  "func.func"() ({}) {function_type = () -> (), 
                      memref = memref<*xf32, 4>,
                      sym_name = "memref"} : () -> ()

  // CHECK: memref<*xf32, 4 : i64>


  "func.func"() ({}) {function_type = () -> (), 
                      dense_resource = dense_resource<resource_1> : tensor<1xi32>,
                      sym_name = "dense_resource"} : () -> ()

  // CHECK: dense_resource<resource_1> : tensor<1xi32>

  "func.func"() ({}) {function_type = () -> (), 
                      type_attr = index,
                      sym_name = "memref"} : () -> ()

  // CHECK: "type_attr" = index

  "func.func"() ({}) {function_type = () -> (), 
                      type_attr = !index,
                      sym_name = "memref"} : () -> ()

  // CHECK: "type_attr" = index

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[1, 0x23, -23, -0x21, ?], offset: -3>,
                      sym_name = "strided"} : () -> ()
  // CHECK: "strided" = strided<[1, 35, -23, -33, ?], offset: -3>

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[], offset: ?>,
                      sym_name = "strided"} : () -> ()
  // CHECK: "strided" = strided<[], offset: ?>

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[], offset: 0>,
                      sym_name = "strided"} : () -> ()
  // CHECK: "strided" = strided<[]>

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[]>,
                      sym_name = "strided"} : () -> ()
  // CHECK: "strided" = strided<[]>

  "func.func"() ({}) {function_type = () -> (),
                      complex = complex<i32>,
                      sym_name = "complex"} : () -> ()
  // CHECK: "complex" = complex<i32>

  "func.func"() ({}) {function_type = () -> (),
                      complex = complex<f32>,
                      sym_name = "complex"} : () -> ()
  // CHECK: "complex" = complex<f32>

}) : () -> ()
