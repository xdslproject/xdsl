// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

hw.module @module() {
  hw.output
}

%foo = "test.op"() : () -> (i32)
hw.instance "too_many_arg" @module(unknownArg: %foo: i32) -> ()

// CHECK: Unknown input port 'unknownArg'

// -----

hw.module @module() {
  hw.output
}

%res = hw.instance "too_many_res" @module() -> (unknownRes: i32)

// CHECK: Unknown output port 'unknownRes'

// -----

hw.module @module(in %foo: i32) {
  hw.output
}

hw.instance "missing_arg" @module() -> ()

// CHECK: Missing input port 'foo'

// -----

hw.module @module(out foo: i32) {
  %foo = arith.constant 0 : i32
  hw.output %foo : i32
}

hw.instance "missing_res" @module() -> ()

// CHECK: Missing output port 'foo'

// -----

hw.module @module(in %foo: i32) {
  hw.output
}

%foo = "test.op"() : () -> (i32)
hw.instance "arg_redef" @module(foo: %foo: i32, foo: %foo: i32) -> ()

// CHECK: Multiple definitions for input port 'foo'

// -----

hw.module @module(out foo: i32) {
  %foo = arith.constant 0 : i32
  hw.output %foo : i32
}

%res1, %res2 = hw.instance "res_redef" @module() -> (foo: i32, foo: i32)

// CHECK: Multiple definitions for output port 'foo'

// -----

hw.instance "no_module" @module() -> ()

// CHECK: Module @module not found

// -----

func.func @module() {
  return
}

hw.instance "bad_module" @module() -> ()

// CHECK: Module @module must be a HWModuleLike, found 'func.func'

// -----

hw.module @module() {
  hw.output
}

"hw.instance"() {"instanceName" = "unexcpected_arg", "moduleName" = @module, "argNames" = ["unexpected"], "resultNames" = []} : () -> ()

// CHECK: Instance has a different amount of argument names (1) and arguments (0)

// -----

hw.module @module() {
  hw.output
}

"hw.instance"() {"instanceName" = "unexpected_res", "moduleName" = @module, "argNames" = [], "resultNames" = ["unexpected"]} : () -> ()

// CHECK: Instance has a different amount of result names (1) and results (0)
