// RUN: xdsl-opt -p canonicalize %s | filecheck %s
%v0, %v1 = "test.op"() : () -> (index, index)

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c3 = arith.constant 3 : index

// CHECK:       builtin.module {
// CHECK-NEXT:    %v0, %v1 = "test.op"() : () -> (index, index)

%r00, %r01 = scf.for %i0 = %v0 to %v1 step %v0 iter_args(%arg00 = %v0, %arg01 = %v1) -> (index, index) {
    "test.op"() {"not constant"} : () -> ()
    yield %arg00, %arg01 : index, index
}
"test.op"(%r00, %r01) : (index, index) -> ()

// CHECK-NEXT:    %r00, %r01 = scf.for %i0 = %v0 to %v1 step %v0 iter_args(%arg00 = %v0, %arg01 = %v1) -> (index, index) {
// CHECK-NEXT:        "test.op"() {"not constant"} : () -> ()
// CHECK-NEXT:      scf.yield %arg00, %arg01 : index, index
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%r00, %r01) : (index, index) -> ()

%r10, %r11 = scf.for %i0 = %c1 to %c1 step %v0 iter_args(%arg10 = %v0, %arg11 = %v1) -> (index, index) {
    "test.op"() {"same bounds"} : () -> ()
    yield %arg10, %arg11 : index, index
}
"test.op"(%r10, %r11) : (index, index) -> ()

// CHECK-NEXT:    "test.op"(%v0, %v1) : (index, index) -> ()

%r20, %r21 = scf.for %i0 = %c2 to %c1 step %v0 iter_args(%arg20 = %v0, %arg21 = %v1) -> (index, index) {
    "test.op"() {"lb > ub"} : () -> ()
    yield %arg20, %arg21 : index, index
}
"test.op"(%r20, %r21) : (index, index) -> ()

// CHECK-NEXT:    "test.op"(%v0, %v1) : (index, index) -> ()

%r30, %r31 = scf.for %i0 = %c1 to %c3 step %c2 iter_args(%arg30 = %v0, %arg31 = %v1) -> (index, index) {
    "test.op"() {"exactly once"} : () -> ()
    yield %arg30, %arg31 : index, index
}
"test.op"(%r30, %r31) : (index, index) -> ()

// CHECK-NEXT:    "test.op"() {"exactly once"} : () -> ()
// CHECK-NEXT:    "test.op"(%v0, %v1) : (index, index) -> ()

// CHECK:       %const = arith.constant 0 : i32
// CHECK-NEXT:  scf.for %i = %v0 to %v1 step %v0 {
// CHECK-NEXT:    "test.op"(%const) : (i32) -> ()
// CHECK-NEXT:  }

scf.for %i = %v0 to %v1 step %v0 {
    %const = arith.constant 0: i32
    "test.op"(%const) : (i32) -> ()
}

// CHECK:       %inner_step = arith.constant 10 : index
// CHECK-NEXT:  %const_1 = arith.constant 0 : i32
// CHECK-NEXT:  scf.for %i_1 = %v0 to %v1 step %v0 {
// CHECK-NEXT:    scf.for %j = %i_1 to %v1 step %inner_step {
// CHECK-NEXT:      "test.op"(%const_1) : (i32) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }

scf.for %i = %v0 to %v1 step %v0 {
    %inner_step = arith.constant 10: index
    scf.for %j = %i to %v1 step %inner_step {
        %const = arith.constant 0: i32
        "test.op"(%const) : (i32) -> ()
    } 
}

// CHECK:      func.func @execute_region() -> i32 {
// CHECK-NEXT:   %a = "test.op"() : () -> i32
// CHECK-NEXT:   %b = arith.constant 1 : i32
// CHECK-NEXT:   %c = arith.addi %a, %b : i32
// CHECK-NEXT:   func.return %c : i32
// CHECK-NEXT: }

func.func @execute_region() -> i32 {
  %a = "test.op"() : () -> (i32)
  %d = scf.execute_region -> (i32) {
    %b = arith.constant 1 : i32
    %c = arith.addi %a, %b : i32
    scf.yield %c : i32
  }
  func.return %d : i32
}

// CHECK:      func.func @execute_region_with_multiple_blocks() -> i32 {
// CHECK-NEXT:   %a, %b = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:   %d = scf.execute_region -> (i32) {
// CHECK-NEXT:     %cond = "test.op"() : () -> i1
// CHECK-NEXT:     cf.cond_br %cond, ^bb0, ^bb1
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     scf.yield %a : i32
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     scf.yield %b : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %d : i32
// CHECK-NEXT: }

func.func @execute_region_with_multiple_blocks() -> i32 {
  %a, %b = "test.op"() : () -> (i32, i32)
  %d = scf.execute_region -> (i32) {
      %cond = "test.op"() : () -> (i1)
      cf.cond_br %cond, ^bb0, ^bb1
    ^bb0:
      scf.yield %a : i32
    ^bb1:
      scf.yield %b : i32
  }
  func.return %d : i32
}

// CHECK-LABEL: func.func @test_true
// CHECK-NOT: scf.if
// CHECK: %{{.*}} = arith.constant 1 : i32
func.func @test_true() -> i32 {
  %true = arith.constant true
  %0 = scf.if %true -> (i32) {
    %1 = arith.constant 1 : i32
    scf.yield %1 : i32
  } else {
    %2 = arith.constant 2 : i32
    scf.yield %2 : i32
  }
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_false
// CHECK-NOT: scf.if
// CHECK: %{{.*}} = arith.constant 2 : i32
func.func @test_false() -> i32 {
  %false = arith.constant false
  %0 = scf.if %false -> (i32) {
    %1 = arith.constant 1 : i32
    scf.yield %1 : i32
  } else {
    %2 = arith.constant 2 : i32
    scf.yield %2 : i32
  }
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_true_void_both_branches
// CHECK-NOT: scf.if
// CHECK: "test.op"() {then = true} : () -> ()
// CHECK-NOT: "test.op"() {else = true}
// CHECK-NEXT: func.return
func.func @test_true_void_both_branches() {
  %true = arith.constant true
  scf.if %true {
    "test.op"() {"then" = true} : () -> ()
  } else {
    "test.op"() {"else" = true} : () -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @test_false_void_both_branches
// CHECK-NOT: scf.if
// CHECK: "test.op"() {else = true} : () -> ()
// CHECK-NOT: "test.op"() {then = true}
// CHECK-NEXT: func.return
func.func @test_false_void_both_branches() {
  %false = arith.constant false
  scf.if %false {
    "test.op"() {"then" = true} : () -> ()
  } else {
    "test.op"() {"else" = true} : () -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @test_if_true_no_else
// CHECK-NOT: scf.if
// CHECK: "test.op"() {value = 10 : i32} : () -> ()
// CHECK-NEXT: func.return
func.func @test_if_true_no_else() {
  %true = arith.constant true
  scf.if %true {
    "test.op"() {"value" = 10 : i32} : () -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @test_if_false_no_else
// CHECK-NOT: scf.if
// CHECK-NEXT: func.return
// CHECK-NOT: test.op
func.func @test_if_false_no_else() {
  %false = arith.constant false
  scf.if %false {
    "test.op"() {"value" = 99 : i32} : () -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @test_empty_then_region_false
// CHECK-NOT: scf.if
// CHECK-NEXT: func.return
func.func @test_empty_then_region_false() {
  %false = arith.constant false
  scf.if %false {
  }
  func.return
}

// CHECK-LABEL: func.func @test_both_empty_regions_true
// CHECK-NOT: scf.if
// CHECK-NEXT: func.return
func.func @test_both_empty_regions_true() {
  %true = arith.constant true
  scf.if %true {
  } else {
  }
  func.return
}

// CHECK-LABEL: func.func @test_both_empty_regions_false
// CHECK-NOT: scf.if
// CHECK-NEXT: func.return
func.func @test_both_empty_regions_false() {
  %false = arith.constant false
  scf.if %false {
  } else {
  }
  func.return
}

// CHECK-LABEL: func.func @test_true_multiple_args
// CHECK-NOT: scf.if
// CHECK: %{{.*}} = arith.constant 1 : i32
// CHECK: %{{.*}} = arith.constant 3 : i32
func.func @test_true_multiple_args() -> (i32, i32) {
  %true = arith.constant true
  %0, %1 = scf.if %true -> (i32, i32) {
    %val1 = arith.constant 1 : i32
    %val2 = arith.constant 3 : i32
    scf.yield %val1, %val2 : i32, i32
  } else {
    %val3 = arith.constant 2 : i32
    %val4 = arith.constant 4 : i32
    scf.yield %val3, %val4 : i32, i32
  }
  func.return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @test_false_multiple_args
// CHECK-NOT: scf.if
// CHECK: %{{.*}} = arith.constant 2 : i32
// CHECK: %{{.*}} = arith.constant 4 : i32
func.func @test_false_multiple_args() -> (i32, i32) {
  %false = arith.constant false
  %0, %1 = scf.if %false -> (i32, i32) {
    %val1 = arith.constant 1 : i32
    %val2 = arith.constant 3 : i32
    scf.yield %val1, %val2 : i32, i32
  } else {
    %val3 = arith.constant 2 : i32
    %val4 = arith.constant 4 : i32
    scf.yield %val3, %val4 : i32, i32
  }
  func.return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @test_true_args_from_outside
// CHECK-NOT: scf.if
// CHECK-NEXT: return %arg0
func.func @test_true_args_from_outside(%arg0: i32, %arg1: i32) -> i32 {
  %true = arith.constant true
  %0 = scf.if %true -> (i32) {
    scf.yield %arg0 : i32
  } else {
    scf.yield %arg1 : i32
  }
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_false_args_from_outside
// CHECK-NOT: scf.if
// CHECK-NEXT: return %arg1
func.func @test_false_args_from_outside(%arg0: i32, %arg1: i32) -> i32 {
  %false = arith.constant false
  %0 = scf.if %false -> (i32) {
    scf.yield %arg0 : i32
  } else {
    scf.yield %arg1 : i32
  }
  func.return %0 : i32
}

// CHECK-LABEL: func.func @test_true_capture_ops
// CHECK-NOT: scf.if
// CHECK: %[[VAL0:.*]] = "test.op"() : () -> i32
// CHECK: "test.op"(%[[VAL0]]) : (i32) -> ()
func.func @test_true_capture_ops() {
  %true = arith.constant true
  %0 = "test.op"() : () -> i32
  scf.if %true {
    "test.op"(%0) : (i32) -> ()
  } else {
    "test.op"() {"else" = true} : () -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @test_false_capture_ops
// CHECK-NOT: scf.if
// CHECK: %[[VAL0:.*]] = "test.op"() : () -> i32
// CHECK: "test.op"() {else = true} : () -> ()
func.func @test_false_capture_ops() {
  %false = arith.constant false
  %0 = "test.op"() : () -> i32
  scf.if %false {
    "test.op"(%0) : (i32) -> ()
  } else {
    "test.op"() {"else" = true} : () -> ()
  }
  func.return
}

// CHECK-LABEL: func.func @test_true_capture_and_yield
// CHECK-NOT: scf.if
// CHECK: %[[VAL0:.*]] = "test.op"() : () -> i32
// CHECK: %[[VAL1:.*]] = arith.addi %[[VAL0]], %[[VAL0]] : i32
// CHECK: return %[[VAL1]] : i32
func.func @test_true_capture_and_yield() -> i32 {
  %true = arith.constant true
  %0 = "test.op"() : () -> i32
  %res = scf.if %true -> (i32) {
    %1 = arith.addi %0, %0 : i32
    scf.yield %1 : i32
  } else {
    scf.yield %0 : i32
  }
  func.return %res : i32
}
