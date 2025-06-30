// RUN: XDSL_ROUNDTRIP

builtin.module {

  // Type tests
  func.func private @struct_to_struct(%0 : !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
    func.return %0 : !llvm.struct<(i32)>
  }

// CHECK:       func.func private @struct_to_struct(%0 : !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
// CHECK-NEXT:    func.return %0 : !llvm.struct<(i32)>
// CHECK-NEXT:  }


  func.func private @struct_to_struct2(%0 : !llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)> {
    func.return %0 : !llvm.struct<(i32, i32)>
  }

// CHECK:       func.func private @struct_to_struct2(%0 : !llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)> {
// CHECK-NEXT:    func.return %0 : !llvm.struct<(i32, i32)>
// CHECK-NEXT:  }

  func.func private @nested_struct_to_struct(%0 : !llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)> {
    func.return %0 : !llvm.struct<(!llvm.struct<(i32)>)>
  }

// CHECK:       func.func private @nested_struct_to_struct(%0 : !llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)> {
// CHECK-NEXT:    func.return %0 : !llvm.struct<(!llvm.struct<(i32)>)>
// CHECK-NEXT:  }

  func.func private @nested_struct_to_struct2(%0 : !llvm.struct<(struct<(i32)>)>) -> !llvm.struct<(struct<(i32)>)> {
    func.return %0 : !llvm.struct<(!llvm.struct<(i32)>)>
  }

// CHECK:       func.func private @nested_struct_to_struct2(%0 : !llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)> {
// CHECK-NEXT:    func.return %0 : !llvm.struct<(!llvm.struct<(i32)>)>
// CHECK-NEXT:  }

  func.func private @array(%0 : !llvm.array<2 x i64>) -> !llvm.array<1 x i32> {
    %1 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i32>
    func.return %1 : !llvm.array<1 x i32>
  }

// CHECK:       func.func private @array(%0 : !llvm.array<2 x i64>) -> !llvm.array<1 x i32> {
// CHECK-NEXT:    %1 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i32>
// CHECK-NEXT:    func.return %1 : !llvm.array<1 x i32>
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
    %0 = arith.constant 1 : i32
    %1 = "llvm.mlir.undef"() : () -> !llvm.struct<(i32)>
    %2 = "llvm.insertvalue"(%1, %0) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
    %3 = "llvm.extractvalue"(%2) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>) -> i32
    %4 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
    func.return
  }

// CHECK:       func.func public @main() {
// CHECK-NEXT:      %0 = arith.constant 1 : i32
// CHECK-NEXT:      %1 = "llvm.mlir.undef"() : () -> !llvm.struct<(i32)>
// CHECK-NEXT:      %2 = "llvm.insertvalue"(%1, %0) <{position = array<i64: 0>}> : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
// CHECK-NEXT:      %3 = "llvm.extractvalue"(%2) <{position = array<i64: 0>}> : (!llvm.struct<(i32)>) -> i32
// CHECK-NEXT:      %4 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  %val = "test.op"() : () -> i32

  %fval = llvm.bitcast %val : i32 to f32

// CHECK:      %val = "test.op"() : () -> i32
// CHECK-NEXT: %fval = llvm.bitcast %val : i32 to f32

  %fval2 = llvm.sitofp %val : i32 to f32

// CHECK-NEXT: %fval2 = llvm.sitofp %val : i32 to f32

  %fval3 = llvm.fpext %fval : f32 to f64

// CHECK-NEXT: %fval3 = llvm.fpext %fval : f32 to f64

  llvm.unreachable {my_attr}

// CHECK-NEXT: llvm.unreachable {my_attr}
}
