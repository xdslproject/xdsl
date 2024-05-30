// RUN: xdsl-opt %s -p function-constant-pinning | filecheck %s


func.func @basic() -> i32 {
    %v = "test.op"() {pin_to_constants = [0]} : () -> i32
    func.return %v : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @basic() -> i32 {
// CHECK-NEXT:     %v = "test.op"() : () -> i32
                   // compare the value to the constant we want to specialize for
// CHECK-NEXT:     %0 = arith.constant 0 : i64
// CHECK-NEXT:     %1 = arith.cmpi eq, %v, %0 : i32
// CHECK-NEXT:     %2 = "scf.if"(%1) ({
                     // if they are equal, branch to specialized function
// CHECK-NEXT:       %3 = func.call @basic_pinned() : () -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       scf.yield %v : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
                 // specialized function here
// CHECK-NEXT:   func.func @basic_pinned() -> i32 {
                   // original op is replaced by constant instantiation
// CHECK-NEXT:     %4 = arith.constant 0 : i64
// CHECK-NEXT:     func.return %4 : i32
// CHECK-NEXT:   }


func.func @control_flow() {
    "test.op"() {"before_op"} : () -> ()

    %cond = "test.op"() {"pin_to_constants"= [true]} : () -> i1

    "scf.if"(%cond) ({
        "test.op"() {"inside_if"} : () -> ()
        scf.yield
    }, {
        scf.yield
    }) : (i1) -> ()

    "test.op"() {"after_op"} : () -> ()

    func.return
}


// CHECK-NEXT:   func.func @control_flow() {
// CHECK-NEXT:      "test.op"() {"before_op"} : () -> ()
// CHECK-NEXT:      %cond = "test.op"() : () -> i1
// CHECK-NEXT:      %5 = arith.constant true
// CHECK-NEXT:      %6 = arith.cmpi eq, %cond, %5 : i1
// CHECK-NEXT:      "scf.if"(%6) ({
// CHECK-NEXT:        func.call @control_flow_pinned() : () -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }, {
                      // inline the rest of the function inside the else statement of the specialization block
                      // (there is no early return in MLIR)
// CHECK-NEXT:        "scf.if"(%cond) ({
// CHECK-NEXT:          "test.op"() {"inside_if"} : () -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "test.op"() {"after_op"} : () -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (i1) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
                  // specialized function does not contain operations that happen before the specialized function
                  // is called ("test.op"() {"before_op"})
// CHECK-NEXT:    func.func @control_flow_pinned() {
// CHECK-NEXT:      %7 = arith.constant true
                    // this scf.if can be constant folded by MLIR later on (not done as part of this pass)
// CHECK-NEXT:      "scf.if"(%7) ({
// CHECK-NEXT:        "test.op"() {"inside_if"} : () -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }, {
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (i1) -> ()
// CHECK-NEXT:      "test.op"() {"after_op"} : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


func.func @function_args(%arg0: memref<100xf32>) -> i32 {
    %v = "test.op"() {pin_to_constants = [0]} : () -> i32

    "test.op"(%v, %arg0) : (i32, memref<100xf32>) -> ()

    func.return %v : i32
}


// CHECK-NEXT:   func.func @function_args(%arg0 : memref<100xf32>) -> i32 {
// CHECK-NEXT:     %v_1 = "test.op"() : () -> i32
// CHECK-NEXT:     %8 = arith.constant 0 : i64
// CHECK-NEXT:     %9 = arith.cmpi eq, %v_1, %8 : i32
// CHECK-NEXT:     %10 = "scf.if"(%9) ({
                     // make sure that we forward function args to the specialized function
                     // and weave return values through the generated if/else
// CHECK-NEXT:       %11 = func.call @function_args_pinned(%arg0) : (memref<100xf32>) -> i32
// CHECK-NEXT:       scf.yield %11 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       "test.op"(%v_1, %arg0) : (i32, memref<100xf32>) -> ()
// CHECK-NEXT:       scf.yield %v_1 : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %10 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @function_args_pinned(%12 : memref<100xf32>) -> i32 {
// CHECK-NEXT:     %13 = arith.constant 0 : i64
                   // here the function arg is used
// CHECK-NEXT:     "test.op"(%13, %12) : (i32, memref<100xf32>) -> ()
// CHECK-NEXT:     func.return %13 : i32
// CHECK-NEXT:   }


// test for everything combined:

func.func @control_flow_and_function_args(%arg: i32) -> i32 {
    %cond = "test.op"() {"pin_to_constants"= [true]} : () -> i1

    "scf.if"(%cond) ({
        "test.op"() {"inside_if"} : () -> ()
        scf.yield
    }, {
        scf.yield
    }) : (i1) -> ()

    %rval = "test.op"(%arg) {"after_op"} : (i32) -> i32

    func.return %rval : i32
}

// CHECK-NEXT:   func.func @control_flow_and_function_args(%arg : i32) -> i32 {
// CHECK-NEXT:     %cond_1 = "test.op"() : () -> i1
// CHECK-NEXT:     %14 = arith.constant true
// CHECK-NEXT:     %15 = arith.cmpi eq, %cond_1, %14 : i1
// CHECK-NEXT:     %16 = "scf.if"(%15) ({
// CHECK-NEXT:       %17 = func.call @control_flow_and_function_args_pinned(%arg) : (i32) -> i32
// CHECK-NEXT:       scf.yield %17 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       "scf.if"(%cond_1) ({
// CHECK-NEXT:         "test.op"() {"inside_if"} : () -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }, {
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }) : (i1) -> ()
// CHECK-NEXT:       %rval = "test.op"(%arg) {"after_op"} : (i32) -> i32
// CHECK-NEXT:       scf.yield %rval : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %16 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @control_flow_and_function_args_pinned(%18 : i32) -> i32 {
// CHECK-NEXT:     %19 = arith.constant true
// CHECK-NEXT:     "scf.if"(%19) ({
// CHECK-NEXT:       "test.op"() {"inside_if"} : () -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }, {
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }) : (i1) -> ()
// CHECK-NEXT:     %20 = "test.op"(%18) {"after_op"} : (i32) -> i32
// CHECK-NEXT:     func.return %20 : i32
// CHECK-NEXT:   }


func.func @specialize_multi_case() -> i32 {
    %v = "test.op"() {pin_to_constants = [0, 1]} : () -> i32
    func.return %v : i32
}

// specialization for multiple values gets pretty ugly pretty quickly, but MLIR canonicalize is able
// to clean it up pretty good

// CHECK-NEXT:   func.func @specialize_multi_case() -> i32 {
// CHECK-NEXT:     %v_2 = "test.op"() : () -> i32
// CHECK-NEXT:     %21 = arith.constant 0 : i64
// CHECK-NEXT:     %22 = arith.cmpi eq, %v_2, %21 : i32
// CHECK-NEXT:     %23 = "scf.if"(%22) ({
// CHECK-NEXT:       %24 = func.call @specialize_multi_case_pinned_1() : () -> i32
// CHECK-NEXT:       scf.yield %24 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %25 = arith.constant 1 : i64
// CHECK-NEXT:       %26 = arith.cmpi eq, %v_2, %25 : i32
// CHECK-NEXT:       %27 = "scf.if"(%26) ({
// CHECK-NEXT:         %28 = func.call @specialize_multi_case_pinned() : () -> i32
// CHECK-NEXT:         scf.yield %28 : i32
// CHECK-NEXT:       }, {
// CHECK-NEXT:         scf.yield %v_2 : i32
// CHECK-NEXT:       }) : (i1) -> i32
// CHECK-NEXT:       scf.yield %27 : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %23 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @specialize_multi_case_pinned_1() -> i32 {
                   // this function still carries the old specialization check within it, but MLIR can see that
                   // the branch is never taken, so it's completely removed.
// CHECK-NEXT:     %29 = arith.constant 0 : i64
// CHECK-NEXT:     %30 = arith.constant 1 : i64
// CHECK-NEXT:     %31 = arith.cmpi eq, %29, %30 : i32
// CHECK-NEXT:     %32 = "scf.if"(%31) ({
// CHECK-NEXT:       %33 = func.call @specialize_multi_case_pinned() : () -> i32
// CHECK-NEXT:       scf.yield %33 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       scf.yield %29 : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @specialize_multi_case_pinned() -> i32 {
// CHECK-NEXT:     %34 = arith.constant 1 : i64
// CHECK-NEXT:     func.return %34 : i32
// CHECK-NEXT:   }
