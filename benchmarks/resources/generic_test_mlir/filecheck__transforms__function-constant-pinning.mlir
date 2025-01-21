"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "basic"}> ({
    %5 = "test.op"() {pin_to_constants = [0 : i32]} : () -> i32
    "func.return"(%5) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "control_flow"}> ({
    "test.op"() {before_op} : () -> ()
    %4 = "test.op"() {pin_to_constants = [true]} : () -> i1
    "scf.if"(%4) ({
      "test.op"() {inside_if} : () -> ()
      "scf.yield"() : () -> ()
    }, {
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
    "test.op"() {after_op} : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<100xf32>) -> i32, sym_name = "function_args"}> ({
  ^bb0(%arg1: memref<100xf32>):
    %3 = "test.op"() {pin_to_constants = [0 : i32]} : () -> i32
    "test.op"(%3, %arg1) : (i32, memref<100xf32>) -> ()
    "func.return"(%3) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32) -> i32, sym_name = "control_flow_and_function_args"}> ({
  ^bb0(%arg0: i32):
    %1 = "test.op"() {pin_to_constants = [true]} : () -> i1
    "scf.if"(%1) ({
      "test.op"() {inside_if} : () -> ()
      "scf.yield"() : () -> ()
    }, {
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
    %2 = "test.op"(%arg0) {after_op} : (i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "specialize_multi_case"}> ({
    %0 = "test.op"() {pin_to_constants = [0 : i32, 1 : i32]} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i32, i64, memref<?xf64>) -> f64, sym_name = "external_test", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
