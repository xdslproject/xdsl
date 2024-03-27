// RUN: xdsl-opt %s -p function-specialization | filecheck %s


func.func @basic() -> i32 {
    %v = "test.op"() {specialize_on_vals = [0]} : () -> i32
    func.return %v : i32
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @basic() -> i32 {
// CHECK-NEXT:     %v = "test.op"() : () -> i32
// CHECK-NEXT:     %0 = arith.constant 0 : i64
// CHECK-NEXT:     %1 = arith.cmpi eq, %v, %0 : i32
// CHECK-NEXT:     %2 = "scf.if"(%1) ({
// CHECK-NEXT:       %3 = func.call @basic_specialized() : () -> i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       scf.yield %v : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @basic_specialized() -> i32 {
// CHECK-NEXT:     %4 = arith.constant 0 : i64
// CHECK-NEXT:     func.return %4 : i32
// CHECK-NEXT:   }


func.func @control_flow() {
    "test.op"() {"before_op"} : () -> ()

    %cond = "test.op"() {"specialize_on_vals"= [true]} : () -> i1

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
// CHECK-NEXT:        func.call @control_flow_specialized() : () -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }, {
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
// CHECK-NEXT:    func.func @control_flow_specialized() {
// CHECK-NEXT:      %7 = arith.constant true
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
    %v = "test.op"() {specialize_on_vals = [0]} : () -> i32

    "test.op"(%v, %arg0) : (i32, memref<100xf32>) -> ()

    func.return %v : i32
}


// CHECK-NEXT:   func.func @function_args(%arg0 : memref<100xf32>) -> i32 {
// CHECK-NEXT:     %v_1 = "test.op"() : () -> i32
// CHECK-NEXT:     %8 = arith.constant 0 : i64
// CHECK-NEXT:     %9 = arith.cmpi eq, %v_1, %8 : i32
// CHECK-NEXT:     %10 = "scf.if"(%9) ({
// CHECK-NEXT:       %11 = func.call @function_args_specialized(%arg0) : (memref<100xf32>) -> i32
// CHECK-NEXT:       scf.yield %11 : i32
// CHECK-NEXT:     }, {
// CHECK-NEXT:       "test.op"(%v_1, %arg0) : (i32, memref<100xf32>) -> ()
// CHECK-NEXT:       scf.yield %v_1 : i32
// CHECK-NEXT:     }) : (i1) -> i32
// CHECK-NEXT:     func.return %10 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @function_args_specialized(%12 : memref<100xf32>) -> i32 {
// CHECK-NEXT:     %13 = arith.constant 0 : i64
// CHECK-NEXT:     "test.op"(%13, %12) : (i32, memref<100xf32>) -> ()
// CHECK-NEXT:     func.return %13 : i32
// CHECK-NEXT:   }



func.func @control_flow_and_function_args(%arg: i32) -> i32 {
    %cond = "test.op"() {"specialize_on_vals"= [true]} : () -> i1

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
// CHECK-NEXT:       %17 = func.call @control_flow_and_function_args_specialized(%arg) : (i32) -> i32
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
// CHECK-NEXT:   func.func @control_flow_and_function_args_specialized(%18 : i32) -> i32 {
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
// CHECK-NEXT: }
