// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

"builtin.module"() ({
  "test.op"() { "test" = #hw.inner_name_ref<1> } : () -> ()
}) : () -> ()

// CHECK:  Expected a module and symbol reference

// -----

"builtin.module"() ({
  "test.op"() { "test" = #hw.inner_name_ref<@Foo> } : () -> ()
}) : () -> ()

// CHECK:  Expected a module and symbol reference

// -----

"builtin.module"() ({
  "test.op"() { "test" = #hw.inner_name_ref<@Foo::@Bar::@Baz> } : () -> ()
}) : () -> ()

// CHECK:  Expected a module and symbol reference
