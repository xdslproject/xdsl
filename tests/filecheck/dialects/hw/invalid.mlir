// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

"builtin.module"() ({
  "test.op"() { "test" = #hw.innerNameRef<1> } : () -> ()
}) : () -> ()

// CHECK:  Expected a module and symbol reference

// -----

"builtin.module"() ({
  "test.op"() { "test" = #hw.innerNameRef<@Foo> } : () -> ()
}) : () -> ()

// CHECK:  Expected a module and symbol reference

// -----

"builtin.module"() ({
  "test.op"() { "test" = #hw.innerNameRef<@Foo::@Bar::@Baz> } : () -> ()
}) : () -> ()

// CHECK:  Expected a module and symbol reference

// -----

"builtin.module"() ({
  "test.op"() { "test" = #hw.innerSym[<@x_1,4,foo>, <@y,5,bar>] } : () -> ()
}) : () -> ()

// CHECK:  Expected "public", "private", or "nested"

// -----

hw.module @bad_dir(hgry %a: i1) {
  hw.output
}

// CHECK: invalid port direction

// -----

hw.module @bad_name(out 9auh: i1) {
  %res = "test.op"() : () -> i1
  hw.output %res : i1
}

// CHECK: expected identifier or string literal as port name

// -----

hw.module @bad_param<9: i1>() {
  hw.output
}

// CHECK: expected parameter name

// -----

hw.module @default_values_not_supported<param: i1 = 0>() {
  hw.output
}

// CHECK: default values for parameters are not yet supported

// -----

hw.module @parameter_not_of_type<param: "foo">() {
  hw.output
}

// CHECK: expected type attribute for parameter

// -----

hw.module @double_param<param: i1, param: i1>() {
  hw.output
}

// CHECK: module has two parameters of same name

// -----

"builtin.module"() ({
  "hw.module"() ({
  ^bb0:
    "hw.output"() : () -> ()
  }) {"sym_name" = "too_few_args", "module_type" = !hw.modty<input a : i32>, "parameters" = []} : () -> ()
}) : () -> ()

// CHECK: missing block arguments in module block

// -----

"builtin.module"() ({
  "hw.module"() ({
  ^bb0(%a: i32, %b: i32):
    "hw.output"() : () -> ()
  }) {"sym_name" = "too_many_args", "module_type" = !hw.modty<input a : i32>, "parameters" = []} : () -> ()
}) : () -> ()

// CHECK: too many block arguments in module block

// -----

"builtin.module"() ({
  "hw.module"() ({
  ^bb0(%a: i8):
    "hw.output"() : () -> ()
  }) {"sym_name" = "wrong_arg", "module_type" = !hw.modty<input a : i32>, "parameters" = []} : () -> ()
}) : () -> ()

// CHECK: input-like port #0 has inconsistent type with its matching module block argument (expected i32, block argument is of type i8)

// -----

"builtin.module"() ({
  "hw.module"() ({
    "hw.output"() : () -> ()
  }) {"sym_name" = "bad_port_name", "module_type" = !hw.modty<output 9foo : i32>, "parameters" = []} : () -> ()
}) : () -> ()

// CHECK: expected port name as identifier or string literal

// -----

hw.module @wrong_amount_output(out foo : i8) {
  hw.output
}

// CHECK: wrong amount of output values (expected 1, got 0)

// -----

hw.module @bad_output_type(out foo : i8, out bar : i32) {
  %res = arith.constant 0 : i32
  hw.output %res, %res : i32, i32
}

// CHECK: output #0 is of unexpected type (expected i8, got i32)

// -----

%test = "test.op"() : () -> !hw.array<6xf32>


// CHECK: f32 should be of base attribute integer_type

// -----

%test = "test.op"() : () -> !hw.array<6x9xi32>

// CHECK: Expected one size in hw.array type

// -----

%const = "test.op"() : () -> i19
%const2 = "test.op"() : () -> i81
%array = "hw.array_create"(%const, %const2) : (i19, i81) -> !hw.array<2xi19>

// CHECK : attribute i19 expected from variable 'I', but got i81

// -----

%array = "test.op"() : () -> !hw.array<2xi19>
%index = "test.op"() : () -> i3
%element = hw.array_get %array[%index] : !hw.array<2xi19>, i3

// CHECK: The index (3 bits wide) must be exactly ceil(log2(length(input))) = 1 bits wide

// -----

%test = "hw.array_create"() : () -> !hw.array<9xi32>

// CHECK: incorrect length for range variable:
// CHECK: expected integer >= 1, got 0
