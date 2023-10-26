// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
  "test.op"() { "test" = #hw.inner_name_ref<@Foo::@Bar> } : () -> ()
  // CHECK:  "test.op"() {"test" = #hw.inner_name_ref<@Foo::@Bar>} : () -> ()
}) : () -> ()
