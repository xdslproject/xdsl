// RUN: xdsl-opt %s -p test-add-timers-to-top-level-funcs --split-input-file | filecheck %s

builtin.module {

  // CHECK:      builtin.module {
  // CHECK-NEXT:   func.func @has_timers(%arg0 : i32, %timers : !llvm.ptr) -> i32 {
  // CHECK-NEXT:     %start = func.call @timer_start() : () -> f64
  // CHECK-NEXT:     "test.op"() : () -> ()
  // CHECK-NEXT:     %end = func.call @timer_end(%start) : (f64) -> f64
  // CHECK-NEXT:     "llvm.store"(%end, %timers) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
  // CHECK-NEXT:     func.return %arg0 : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.func private @timer_start() -> f64
  // CHECK-NEXT:   func.func private @timer_end(f64) -> f64
  // CHECK-NEXT: }

  func.func @has_timers(%arg0 : i32, %timers : !llvm.ptr) -> i32 {
    %start = func.call @timer_start() : () -> f64
    "test.op"() : () -> ()
    %end = func.call @timer_end(%start) : (f64) -> f64
    "llvm.store"(%end, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
    func.return %arg0 : i32
  }
  func.func private @timer_start() -> f64
  func.func private @timer_end(f64) -> f64
}

// -----

builtin.module {

  // CHECK:      builtin.module {
  // CHECK-NEXT:   func.func @has_no_timers(%arg0 : i32, %arg1 : i32, %timers : !llvm.ptr) -> i32 {
  // CHECK-NEXT:     %timestamp = func.call @timer_start() : () -> f64
  // CHECK-NEXT:     %res = arith.addi %arg0, %arg1 : i32
  // CHECK-NEXT:     %timediff = func.call @timer_end(%timestamp) : (f64) -> f64
  // CHECK-NEXT:     "llvm.store"(%timediff, %timers) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
  // CHECK-NEXT:     func.return %res : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.func @also_has_no_timers(%timers : !llvm.ptr) {
  // CHECK-NEXT:     %timestamp = func.call @timer_start() : () -> f64
  // CHECK-NEXT:     func.func @nested_should_not_get_timers() {
  // CHECK-NEXT:       func.return
  // CHECK-NEXT:     }
  // CHECK-NEXT:     "test.op"() : () -> ()
  // CHECK-NEXT:     %timediff = func.call @timer_end(%timestamp) : (f64) -> f64
  // CHECK-NEXT:     "llvm.store"(%timediff, %timers) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
  // CHECK-NEXT:     func.return
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.func private @timer_start() -> f64
  // CHECK-NEXT:   func.func private @timer_end(f64) -> f64
  // CHECK-NEXT: }

  func.func @has_no_timers(%arg0 : i32, %arg1 : i32) -> i32 {
    %res = arith.addi %arg0, %arg1 : i32
    func.return %res : i32
  }

  func.func @also_has_no_timers() {
    func.func @nested_should_not_get_timers() {
      func.return
    }
    "test.op"() : () -> ()
    func.return
  }
}
