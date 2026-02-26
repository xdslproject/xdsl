// RUN: xdsl-opt -p scf-for-loop-unroll %s | filecheck %s

// CHECK-LABEL: simple
//      CHECK:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %c2 = arith.constant 2 : index
// CHECK-NEXT:  %c6 = arith.constant 6 : index
// CHECK-NEXT:  %init, %live_in = "test.op"() : () -> (f32, f32)
// CHECK-NEXT:  %i = arith.constant 1 : index
// CHECK-NEXT:  %acc_out = "test.op"(%i, %init, %live_in) : (index, f32, f32) -> f32
// CHECK-NEXT:  %i_1 = arith.constant 3 : index
// CHECK-NEXT:  %acc_out_1 = "test.op"(%i_1, %acc_out, %live_in) : (index, f32, f32) -> f32
// CHECK-NEXT:  %i_2 = arith.constant 5 : index
// CHECK-NEXT:  %acc_out_2 = "test.op"(%i_2, %acc_out_1, %live_in) : (index, f32, f32) -> f32
// CHECK-NEXT:  "test.op"(%acc_out_2) : (f32) -> ()

func.func @simple() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index

    %init, %live_in = "test.op"() : () -> (f32, f32)

    %res = scf.for %i = %c1 to %c6 step %c2 iter_args(%acc_in = %init) -> (f32) {
        %acc_out = "test.op"(%i, %acc_in, %live_in) : (index, f32, f32) -> f32
        scf.yield %acc_out : f32
    }

    "test.op"(%res) : (f32) -> ()
    func.return
}


// CHECK-LABEL: no_loop_iterations_lb_ge_ub
//      CHECK: %c5 = arith.constant 5 : index
// CHECK-NEXT: %c2 = arith.constant 2 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %init = "test.op"() : () -> f32
// CHECK-NEXT: "test.op"(%init) : (f32) -> ()
func.func @no_loop_iterations_lb_ge_ub() {
  %c5 = arith.constant 5 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %init = "test.op"() : () -> f32

  %res = scf.for %i = %c5 to %c1 step %c2 iter_args(%acc = %init) -> (f32) {
    %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
    scf.yield %acc_out : f32
  }
  "test.op"(%res) : (f32) -> ()
  func.return
}

// CHECK-LABEL: func.func @non_constant_lb
//      CHECK: %lb = "test.op"() : () -> index
// CHECK-NEXT: %c10 = arith.constant 10 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %init = "test.op"() : () -> f32
// CHECK-NEXT: %res = scf.for %i = %lb to %c10 step %c1 iter_args(%acc = %init) -> (f32) {
// CHECK-NEXT:   %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
// CHECK-NEXT:   scf.yield %acc_out : f32
// CHECK-NEXT: }
// CHECK-NEXT: "test.op"(%res) : (f32) -> ()
func.func @non_constant_lb() {
  %lb = "test.op"() : () -> index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %init = "test.op"() : () -> f32

  %res = scf.for %i = %lb to %c10 step %c1 iter_args(%acc = %init) -> (f32) {
    %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
    scf.yield %acc_out : f32
  }
  "test.op"(%res) : (f32) -> ()
  func.return
}

// CHECK-LABEL: func.func @non_constant_ub
//      CHECK: %c0 = arith.constant 0 : index
// CHECK-NEXT: %ub = "test.op"() : () -> index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %init = "test.op"() : () -> f32
// CHECK-NEXT: %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (f32) {
// CHECK-NEXT:   %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
// CHECK-NEXT:   scf.yield %acc_out : f32
// CHECK-NEXT: }
// CHECK-NEXT: "test.op"(%res) : (f32) -> ()
func.func @non_constant_ub() {
  %c0 = arith.constant 0 : index
  %ub = "test.op"() : () -> index
  %c1 = arith.constant 1 : index
  %init = "test.op"() : () -> f32

  %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (f32) {
    %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
    scf.yield %acc_out : f32
  }
  "test.op"(%res) : (f32) -> ()
  func.return
}

// CHECK-LABEL: func.func @non_constant_step
//      CHECK: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c10 = arith.constant 10 : index
// CHECK-NEXT: %step = "test.op"() : () -> index
// CHECK-NEXT: %init = "test.op"() : () -> f32
// CHECK-NEXT: %res = scf.for %i = %c0 to %c10 step %step iter_args(%acc = %init) -> (f32) {
// CHECK-NEXT:   %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
// CHECK-NEXT:   scf.yield %acc_out : f32
// CHECK-NEXT: }
// CHECK-NEXT: "test.op"(%res) : (f32) -> ()
func.func @non_constant_step() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %step = "test.op"() : () -> index
  %init = "test.op"() : () -> f32

  %res = scf.for %i = %c0 to %c10 step %step iter_args(%acc = %init) -> (f32) {
    %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
    scf.yield %acc_out : f32
  }
  "test.op"(%res) : (f32) -> ()
  func.return
}

// CHECK-LABEL: func.func @step_does_not_divide_range_evenly
//      CHECK: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c5 = arith.constant 5 : index
// CHECK-NEXT: %c2 = arith.constant 2 : index
// CHECK-NEXT: %init = "test.op"() : () -> f32
// CHECK-NEXT: %i = arith.constant 0 : index
// CHECK-NEXT: %acc_out = "test.op"(%i, %init) : (index, f32) -> f32
// CHECK-NEXT: %i_1 = arith.constant 2 : index
// CHECK-NEXT: %acc_out_1 = "test.op"(%i_1, %acc_out) : (index, f32) -> f32
// CHECK-NEXT: %i_2 = arith.constant 4 : index
// CHECK-NEXT: %acc_out_2 = "test.op"(%i_2, %acc_out_1) : (index, f32) -> f32
// CHECK-NEXT: "test.op"(%acc_out_2) : (f32) -> ()
func.func @step_does_not_divide_range_evenly() {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c2 = arith.constant 2 : index
  %init = "test.op"() : () -> f32

  %res = scf.for %i = %c0 to %c5 step %c2 iter_args(%acc = %init) -> (f32) {
    %acc_out = "test.op"(%i, %acc) : (index, f32) -> f32
    scf.yield %acc_out : f32
  }
  "test.op"(%res) : (f32) -> ()
  func.return
}

// CHECK-LABEL: func.func @no_iter_args
//      CHECK: %c0 = arith.constant 0 : index
// CHECK-NEXT: %c3 = arith.constant 3 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: %i = arith.constant 0 : index
// CHECK-NEXT: "test.op"(%i) : (index) -> ()
// CHECK-NEXT: %i_1 = arith.constant 1 : index
// CHECK-NEXT: "test.op"(%i_1) : (index) -> ()
// CHECK-NEXT: %i_2 = arith.constant 2 : index
// CHECK-NEXT: "test.op"(%i_2) : (index) -> ()
func.func @no_iter_args() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c3 step %c1 {
    "test.op"(%i) : (index) -> ()
  }
  func.return
}
