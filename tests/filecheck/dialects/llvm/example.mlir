// RUN: XDSL_ROUNDTRIP

builtin.module {

  // Type tests
  func.func private @struct_to_struct(%0 : !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
    func.return %0 : !llvm.struct<(i32)>
  }

// CHECK:       func.func private @struct_to_struct(%0 : !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
// CHECK-NEXT:    func.return %0 : !llvm.struct<(i32)>
// CHECK-NEXT:  }


  func.func private @struct_to_struct2(%1 : !llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)> {
    func.return %1 : !llvm.struct<(i32, i32)>
  }

// CHECK:       func.func private @struct_to_struct2(%1 : !llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)> {
// CHECK-NEXT:    func.return %1 : !llvm.struct<(i32, i32)>
// CHECK-NEXT:  }

  func.func private @nested_struct_to_struct(%2 : !llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)> {
    func.return %2 : !llvm.struct<(!llvm.struct<(i32)>)>
  }

// CHECK:       func.func private @nested_struct_to_struct(%2 : !llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)> {
// CHECK-NEXT:    func.return %2 : !llvm.struct<(!llvm.struct<(i32)>)>
// CHECK-NEXT:  }

  func.func private @array(%3 : !llvm.array<2 x i64>) -> !llvm.array<1 x i32> {
    %4 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i32>
    func.return %4 : !llvm.array<1 x i32>
  }

// CHECK:       func.func private @array(%3 : !llvm.array<2 x i64>) -> !llvm.array<1 x i32> {
// CHECK-NEXT:    %4 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i32>
// CHECK-NEXT:    func.return %4 : !llvm.array<1 x i32>
// CHECK-NEXT:  }

  // literal
  %s0, %s1 = "test.op"() : () -> (!llvm.struct<()>, !llvm.struct<(i32)>)
  // named
  %s2, %s3 = "test.op"() : () -> (!llvm.struct<"a", ()>, !llvm.struct<"a", (i32)>)

  // CHECK-NEXT:   %s0, %s1 = "test.op"() : () -> (!llvm.struct<()>, !llvm.struct<(i32)>)
  // CHECK-NEXT:   %s2, %s3 = "test.op"() : () -> (!llvm.struct<"a", ()>, !llvm.struct<"a", (i32)>)

  // void
  %v0 = "test.op"() : () -> !llvm.void
  // CHECK-NEXT:   %v0 = "test.op"() : () -> !llvm.void

  // function
  %f0 = "test.op"() : () -> !llvm.func<void ()>
  %f1 = "test.op"() : () -> !llvm.func<i32 (i32, i32)>
  // CHECK-NEXT:   %f0 = "test.op"() : () -> !llvm.func<void ()>
  // CHECK-NEXT:   %f1 = "test.op"() : () -> !llvm.func<i32 (i32, i32)>


  // Op tests
  func.func public @main() {
    %5 = arith.constant 1 : i32
    %6 = "llvm.mlir.undef"() : () -> !llvm.struct<(i32)>
    %7 = "llvm.insertvalue"(%6, %5) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
    %8 = "llvm.extractvalue"(%7) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>) -> i32
    func.return
  }

// CHECK:       func.func public @main() {
// CHECK-NEXT:      %5 = arith.constant 1 : i32
// CHECK-NEXT:      %6 = "llvm.mlir.undef"() : () -> !llvm.struct<(i32)>
// CHECK-NEXT:      %7 = "llvm.insertvalue"(%6, %5) <{"position" = array<i64: 0>}> : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
// CHECK-NEXT:      %8 = "llvm.extractvalue"(%7) <{"position" = array<i64: 0>}> : (!llvm.struct<(i32)>) -> i32
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}
