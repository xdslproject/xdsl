// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @symref_test() {
    // Declare some symbols
    "symref.declare"() {"sym_name" = "counter"} : () -> ()
    "symref.declare"() {"sym_name" = "flag"} : () -> ()

    // Test basic fetch operations
    %0 = "symref.fetch"() {"symbol" = @counter} : () -> i32
    %1 = "symref.fetch"() {"symbol" = @flag} : () -> i1

    // Create some values to update with
    %c42 = arith.constant 42 : i32
    %true = arith.constant true

    // Test update operations
    "symref.update"(%c42) {"symbol" = @counter} : (i32) -> ()
    "symref.update"(%true) {"symbol" = @flag} : (i1) -> ()

    // Test fetch after update
    %3 = "symref.fetch"() {"symbol" = @counter} : () -> i32
    %4 = "symref.fetch"() {"symbol" = @flag} : () -> i1

    func.return
  }

  func.func @symref_complex_types() {
    // Test with more complex types
    "symref.declare"() {"sym_name" = "tensor_ref"} : () -> ()
    "symref.declare"() {"sym_name" = "memref_ref"} : () -> ()

    %tensor = "test.op"() : () -> tensor<4x4xf32>
    %memref = "test.op"() : () -> memref<8xi64>

    "symref.update"(%tensor) {"symbol" = @tensor_ref} : (tensor<4x4xf32>) -> ()
    "symref.update"(%memref) {"symbol" = @memref_ref} : (memref<8xi64>) -> ()

    %7 = "symref.fetch"() {"symbol" = @tensor_ref} : () -> tensor<4x4xf32>
    %8 = "symref.fetch"() {"symbol" = @memref_ref} : () -> memref<8xi64>

    func.return
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func @symref_test() {
// CHECK-NEXT:     symref.declare "counter"
// CHECK-NEXT:     symref.declare "flag"
// CHECK-NEXT:     %{{.*}} = symref.fetch @counter : i32
// CHECK-NEXT:     %{{.*}} = symref.fetch @flag : i1
// CHECK-NEXT:     %{{.*}} = arith.constant 42 : i32
// CHECK-NEXT:     %{{.*}} = arith.constant true
// CHECK-NEXT:     symref.update @counter = %{{.*}} : i32
// CHECK-NEXT:     symref.update @flag = %{{.*}} : i1
// CHECK-NEXT:     %{{.*}} = symref.fetch @counter : i32
// CHECK-NEXT:     %{{.*}} = symref.fetch @flag : i1
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @symref_complex_types() {
// CHECK-NEXT:     symref.declare "tensor_ref"
// CHECK-NEXT:     symref.declare "memref_ref"
// CHECK-NEXT:     %{{.*}} = "test.op"() : () -> tensor<4x4xf32>
// CHECK-NEXT:     %{{.*}} = "test.op"() : () -> memref<8xi64>
// CHECK-NEXT:     symref.update @tensor_ref = %{{.*}} : tensor<4x4xf32>
// CHECK-NEXT:     symref.update @memref_ref = %{{.*}} : memref<8xi64>
// CHECK-NEXT:     %{{.*}} = symref.fetch @tensor_ref : tensor<4x4xf32>
// CHECK-NEXT:     %{{.*}} = symref.fetch @memref_ref : memref<8xi64>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
