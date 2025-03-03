// RUN: xdsl-opt -p convert-func-to-x86-func --verify-diagnostics --split-input-file  %s | filecheck %s

// CHECK: Cannot lower external functions (not implemented)
func.func @foo_int(%0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32, %6: i32, %7: i32) -> i32

// -----

// CHECK: Cannot lower function parameters bigger than 64 bits (not implemented)
func.func @f(%a: i128) -> () {
  func.return
}

// -----

// CHECK: Cannot lower shaped function parameters (not implemented)
func.func @f(%a: memref<1xf32>) -> () {
  func.return
}

// -----

// CHECK: Cannot lower func.return with more than 1 argument (not implemented)
func.func @foo_int() -> (i32,i32) {
  %a = "test.op"(): () -> i32
  func.return %a,%a: i32,i32
}

// -----

// CHECK: Cannot lower function return values bigger than 64 bits (not implemented)
func.func @foo_int() -> (i128) {
  %a = "test.op"(): () -> i128
  func.return %a: i128
}

// -----

// CHECK: Cannot lower shaped function output (not implemented)
func.func @f() -> (memref<1xf32>) {
  %a = "test.op"(): () -> memref<1xf32>
  func.return %a: memref<1xf32>
}
