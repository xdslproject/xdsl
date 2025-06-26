// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @symref_test() {
    // CHECK:     symref.declare "counter"
    // CHECK:     symref.declare "tensor_ref"
    "symref.declare"() {"sym_name" = "counter"} : () -> ()
    "symref.declare"() {"sym_name" = "tensor_ref"} : () -> ()

    // CHECK:     %{{.*}} = symref.fetch @counter : i32
    // CHECK:     %{{.*}} = symref.fetch @tensor_ref : tensor<4x4xf32>
    %0 = "symref.fetch"() {"symbol" = @counter} : () -> i32
    %1 = "symref.fetch"() {"symbol" = @tensor_ref} : () -> tensor<4x4xf32>

    // CHECK:     %{{.*}} = arith.constant 42 : i32
    // CHECK:     %{{.*}} = "test.op"() : () -> tensor<4x4xf32>
    %c42 = arith.constant 42 : i32
    %tensor = "test.op"() : () -> tensor<4x4xf32>

    // CHECK:     symref.update @counter = %{{.*}} : i32
    // CHECK:     symref.update @tensor_ref = %{{.*}} : tensor<4x4xf32>
    "symref.update"(%c42) {"symbol" = @counter} : (i32) -> ()
    "symref.update"(%tensor) {"symbol" = @tensor_ref} : (tensor<4x4xf32>) -> ()

    // CHECK:     %{{.*}} = symref.fetch @counter : i32
    // CHECK:     %{{.*}} = symref.fetch @tensor_ref : tensor<4x4xf32>
    %3 = "symref.fetch"() {"symbol" = @counter} : () -> i32
    %4 = "symref.fetch"() {"symbol" = @tensor_ref} : () -> tensor<4x4xf32>

    func.return
  }
}
