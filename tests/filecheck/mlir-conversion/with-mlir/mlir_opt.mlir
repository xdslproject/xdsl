// Shorthand to use mlir-opt within an xDSL pipeline: will use
// mlir-opt --allow-unregistered-dialect --mlir-print-op-generic -p builtin.pipeline(<input-pipeline>)
// RUN: xdsl-opt %s -p mlir-opt[cse] --print-op-generic | filecheck %s

// The explicit syntax for the MLIROptPass, allowing full control over the used arguments
// Here, using --mlir-disable-threading to disable threading in mlir-opt
// RUN: xdsl-opt %s -p mlir-opt{arguments='--cse','--mlir-disable-threading','--mlir-print-op-generic'} --print-op-generic | filecheck %s

// Here, using the generic argument to use xDSL's custom syntax printing to send IR to
// MLIR.
// RUN: xdsl-opt %s -p mlir-opt{generic=false\ arguments='--cse','--mlir-print-op-generic'} --print-op-generic | filecheck %s
// RUN: xdsl-opt %s -p mlir-opt{generic=true\ arguments='--cse','--mlir-print-op-generic'} --print-op-generic | filecheck %s

// Check that manually passing an executable works
// RUN: xdsl-opt %s -p mlir-opt{executable=mlir-opt\ generic=true\ arguments='--cse','--mlir-print-op-generic'} --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "do_nothing"} : () -> ()
}) {"gpu.container_module"} : () -> ()


// CHECK:         "builtin.module"() ({
// CHECK-NEXT:      "func.func"() <{function_type = () -> (), sym_name = "do_nothing"}> ({
// CHECK-NEXT:        "func.return"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:    }) {gpu.container_module} : () -> ()
