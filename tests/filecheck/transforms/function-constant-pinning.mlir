// RUN: xdsl-opt %s -p function-constant-pinning | filecheck %s


func.func @basic() -> i32 {
    %v = "test.op"() {pin_to_constants = [0 : i32]} : () -> i32
    func.return %v : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @basic() -> i32 {
// CHECK-NEXT:     %v = "test.op"() : () -> i32
                   // compare the value to the constant we want to specialize for
// CHECK-NEXT:     %0 = arith.constant 0 : i32
// CHECK-NEXT:     %1 = arith.cmpi eq, %v, %0 : i32
// CHECK-NEXT:     %2 = scf.if %1 -> (i32) {
                     // if they are equal, branch to specialized function
// CHECK-NEXT:       %3 = func.call @basic_pinned() : () -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %v : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
                 // specialized function here
// CHECK-NEXT:   func.func @basic_pinned() -> i32 {
                   // original op is replaced by constant instantiation
// CHECK-NEXT:     %v = arith.constant 0 : i32
// CHECK-NEXT:     func.return %v : i32
// CHECK-NEXT:   }


func.func @control_flow() {
    "test.op"() {"before_op"} : () -> ()

    %cond = "test.op"() {"pin_to_constants"= [true]} : () -> i1

    scf.if %cond {
        "test.op"() {"inside_if"} : () -> ()
    } else {
      scf.yield
    }

    "test.op"() {"after_op"} : () -> ()

    func.return
}

// CHECK-NEXT:   func.func @control_flow() {
// CHECK-NEXT:      "test.op"() {before_op} : () -> ()
// CHECK-NEXT:      %cond = "test.op"() : () -> i1
// CHECK-NEXT:      %0 = arith.constant true
// CHECK-NEXT:      %1 = arith.cmpi eq, %cond, %0 : i1
// CHECK-NEXT:      scf.if %1 {
// CHECK-NEXT:        func.call @control_flow_pinned() : () -> ()
// CHECK-NEXT:      } else {
                      // inline the rest of the function inside the else statement of the specialization block
                      // (there is no early return in MLIR)
// CHECK-NEXT:        scf.if %cond {
// CHECK-NEXT:          "test.op"() {inside_if} : () -> ()
// CHECK-NEXT:        } else {
// CHECK-NEXT:        }
// CHECK-NEXT:        "test.op"() {after_op} : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
                  // specialized function does not contain operations that happen before the specialized function
                  // is called ("test.op"() {before_op})
// CHECK-NEXT:    func.func @control_flow_pinned() {
// CHECK-NEXT:      %cond = arith.constant true
                    // this scf.if can be constant folded by MLIR later on (not done as part of this pass)
// CHECK-NEXT:      scf.if %cond {
// CHECK-NEXT:        "test.op"() {inside_if} : () -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:      }
// CHECK-NEXT:      "test.op"() {after_op} : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


func.func @function_args(%arg0: memref<100xf32>) -> i32 {
    %v = "test.op"() {pin_to_constants = [0 : i32]} : () -> i32

    "test.op"(%v, %arg0) : (i32, memref<100xf32>) -> ()

    func.return %v : i32
}


// CHECK-NEXT:   func.func @function_args(%arg0 : memref<100xf32>) -> i32 {
// CHECK-NEXT:     %v = "test.op"() : () -> i32
// CHECK-NEXT:     %0 = arith.constant 0 : i32
// CHECK-NEXT:     %1 = arith.cmpi eq, %v, %0 : i32
// CHECK-NEXT:     %2 = scf.if %1 -> (i32) {
                     // make sure that we forward function args to the specialized function
                     // and weave return values through the generated if/else
// CHECK-NEXT:       %3 = func.call @function_args_pinned(%arg0) : (memref<100xf32>) -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       "test.op"(%v, %arg0) : (i32, memref<100xf32>) -> ()
// CHECK-NEXT:       scf.yield %v : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @function_args_pinned(%arg0 : memref<100xf32>) -> i32 {
// CHECK-NEXT:     %v = arith.constant 0 : i32
                   // here the function arg is used
// CHECK-NEXT:     "test.op"(%v, %arg0) : (i32, memref<100xf32>) -> ()
// CHECK-NEXT:     func.return %v : i32
// CHECK-NEXT:   }


// test for everything combined:

func.func @control_flow_and_function_args(%arg: i32) -> i32 {
    %cond = "test.op"() {"pin_to_constants"= [true]} : () -> i1

    scf.if %cond {
        "test.op"() {"inside_if"} : () -> ()
    } else {
        scf.yield
    }

    %rval = "test.op"(%arg) {"after_op"} : (i32) -> i32

    func.return %rval : i32
}

// CHECK-NEXT:   func.func @control_flow_and_function_args(%arg : i32) -> i32 {
// CHECK-NEXT:     %cond = "test.op"() : () -> i1
// CHECK-NEXT:     %0 = arith.constant true
// CHECK-NEXT:     %1 = arith.cmpi eq, %cond, %0 : i1
// CHECK-NEXT:     %2 = scf.if %1 -> (i32) {
// CHECK-NEXT:       %3 = func.call @control_flow_and_function_args_pinned(%arg) : (i32) -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.if %cond {
// CHECK-NEXT:         "test.op"() {inside_if} : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:       }
// CHECK-NEXT:       %rval = "test.op"(%arg) {after_op} : (i32) -> i32
// CHECK-NEXT:       scf.yield %rval : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @control_flow_and_function_args_pinned(%arg : i32) -> i32 {
// CHECK-NEXT:     %cond = arith.constant true
// CHECK-NEXT:     scf.if %cond {
// CHECK-NEXT:       "test.op"() {inside_if} : () -> ()
// CHECK-NEXT:     } else {
// CHECK-NEXT:     }
// CHECK-NEXT:     %rval = "test.op"(%arg) {after_op} : (i32) -> i32
// CHECK-NEXT:     func.return %rval : i32
// CHECK-NEXT:   }


func.func @specialize_multi_case() -> i32 {
    %v = "test.op"() {pin_to_constants = [0 : i32, 1 : i32]} : () -> i32
    func.return %v : i32
}

// specialization for multiple values gets pretty ugly pretty quickly, but MLIR canonicalize is able
// to clean it up pretty good

// CHECK-NEXT:   func.func @specialize_multi_case() -> i32 {
// CHECK-NEXT:     %v = "test.op"() : () -> i32
// CHECK-NEXT:     %0 = arith.constant 0 : i32
// CHECK-NEXT:     %1 = arith.cmpi eq, %v, %0 : i32
// CHECK-NEXT:     %2 = scf.if %1 -> (i32) {
// CHECK-NEXT:       %3 = func.call @specialize_multi_case_pinned_1() : () -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %4 = arith.constant 1 : i32
// CHECK-NEXT:       %5 = arith.cmpi eq, %v, %4 : i32
// CHECK-NEXT:       %6 = scf.if %5 -> (i32) {
// CHECK-NEXT:         %7 = func.call @specialize_multi_case_pinned() : () -> i32
// CHECK-NEXT:         scf.yield %7 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %v : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %6 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @specialize_multi_case_pinned_1() -> i32 {
                   // this function still carries the old specialization check within it, but MLIR can see that
                   // the branch is never taken, so it's completely removed.
// CHECK-NEXT:     %v = arith.constant 0 : i32
// CHECK-NEXT:     %0 = arith.constant 1 : i32
// CHECK-NEXT:     %1 = arith.cmpi eq, %v, %0 : i32
// CHECK-NEXT:     %2 = scf.if %1 -> (i32) {
// CHECK-NEXT:       %3 = func.call @specialize_multi_case_pinned() : () -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %v : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @specialize_multi_case_pinned() -> i32 {
// CHECK-NEXT:     %v = arith.constant 1 : i32
// CHECK-NEXT:     func.return %v : i32
// CHECK-NEXT:   }

func.func private @external_test(i32, i64, memref<?xf64>) -> f64
// CHECK-NEXT: func.func private @external_test(i32, i64, memref<?xf64>) -> f64
