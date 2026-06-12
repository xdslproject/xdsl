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
// RUN: xdsl-opt %s -p mlir-opt{executable=$XDSL_MLIR_OPT\ generic=true\ arguments='--cse','--mlir-print-op-generic'} --print-op-generic | filecheck %s

module attributes {gpu.container_module} {
  func.func @do_nothing() -> () {
    %0 = arith.constant 1 : i32
    return
  }
}

// CHECK:         "builtin.module"() ({
// CHECK-NEXT:      "func.func"() <{function_type = () -> (), sym_name = "do_nothing"}> ({
// CHECK-NEXT:        "func.return"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:    }) {gpu.container_module} : () -> ()

// Check executables

// Neither env var nor custom executable param
// RUN: xdsl-opt %s -p mlir-opt | filecheck %s --check-prefix CHECK-NEITHER

// CHECK-NEITHER: @do_nothing

// Env var but not custom executable param
// RUN: env XDSL_MLIR_OPT=%S/mlir_opt_foo.sh xdsl-opt %s -p mlir-opt | filecheck %s --check-prefix CHECK-ENV

// CHECK-ENV: @foo

// Custom executable param but not env var
// RUN: xdsl-opt %s -p mlir-opt{executable='"%S/mlir_opt_bar.sh"'} | filecheck %s --check-prefix CHECK-PARAM

// CHECK-PARAM: @bar

// Custom executable param but not env var
// RUN: env XDSL_MLIR_OPT=%S/mlir_opt_foo.sh xdsl-opt %s -p mlir-opt{executable='"%S/mlir_opt_bar.sh"'} | filecheck %s --check-prefix CHECK-BOTH

// CHECK-BOTH: @bar
