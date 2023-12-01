// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

"builtin.module"() ({
  "test.op"() { "test" = #hw.inner_name_ref<@Foo::@Bar> } : () -> ()
  // CHECK:  "test.op"() {"test" = #hw.inner_name_ref<@Foo::@Bar>} : () -> ()
}) : () -> ()

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "test.op"() {"test" = #hw.inner_name_ref<@Foo::@Bar>} : () -> ()
// CHECK-GENERIC-NEXT: })
