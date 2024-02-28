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
