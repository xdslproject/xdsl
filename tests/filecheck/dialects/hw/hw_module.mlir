// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// Custom format input tests

// CHECK: hw.module @custom_basic(in %{{[^ ]*}} a: i1, out nameOfPortInSV: i1, out "": i1, in %{{[^ ]*}} customName: i1, in %{{[^ ]*}} "very custom name": i32) {
// CHECK-NEXT: hw.output %{{[^ ]*}}, %{{[^ ]*}} : i1, i1
// CHECK-NEXT: }
// CHECK-GENERIC: "hw.module"() ({
// CHECK-GENERIC-NEXT: ^{{[^(]*}}(%{{[^ ]*}} : i1, %{{[^ ]*}} : i1, %{{[^ ]*}} : i32):
// CHECK-GENERIC-NEXT:   "hw.output"(%{{[^ ]*}}, %{{[^ ]*}}) : (i1, i1) -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "custom_basic", module_type = !hw.modty<input a : i1, output nameOfPortInSV : i1, output "" : i1, input customName : i1, input "very custom name" : i32>, parameters = []} : () -> ()
hw.module @custom_basic(in %a_foo a: i1, out nameOfPortInSV: i1, out "": i1, in %b customName: i1, in %c "very custom name": i32) {
  hw.output %a_foo, %b: i1, i1
}

// CHECK: hw.module @custom_param<p1: i42, "p2 with wack name": i1>() {
// CHECK-NEXT: hw.output
// CHECK-NEXT: }
// CHECK-GENERIC: "hw.module"() ({
// CHECK-GENERIC-NEXT:   "hw.output"() : () -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "custom_param", module_type = !hw.modty<>, parameters = [#hw.param.decl<"p1": i42>, #hw.param.decl<"p2 with wack name": i1>]} : () -> ()
hw.module @custom_param<p1: i42, "p2 with wack name": i1>() {
  hw.output
}

// CHECK: hw.module private @private_module() {
// CHECK-NEXT: hw.output
// CHECK-NEXT: }
// CHECK-GENERIC: "hw.module"() ({
// CHECK-GENERIC-NEXT: "hw.output"() : () -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "private_module", module_type = !hw.modty<>, parameters = [], sym_visibility = "private"} : () -> ()
hw.module private @private_module() {
  hw.output
}

// CHECK: hw.module @"wack name!!"() {
// CHECK-NEXT: hw.output
// CHECK-NEXT: }
// CHECK-GENERIC: "hw.module"() ({
// CHECK-GENERIC-NEXT:   "hw.output"() : () -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "wack name!!", module_type = !hw.modty<>, parameters = []} : () -> ()
hw.module @"wack name!!"() {
  hw.output
}

// CHECK: hw.module @name_preserve(in %preserve: i8)
// CHECK-GENERIC: {sym_name = "name_preserve", module_type = !hw.modty<input preserve : i8>, parameters = []} : () -> ()
hw.module @name_preserve(in %preserve: i8) {
  hw.output
}

// CHECK: hw.module @more_attrs() attributes {foo} {
// CHECK-GENERIC: }) {sym_name = "more_attrs", module_type = !hw.modty<>, parameters = [], foo} : () -> ()
hw.module @more_attrs() attributes {"foo"} {
  hw.output
}

// CHECK: hw.module.extern @extern(in %{{.*}} foo: i32, in %{{.*}} hello: i16, out baz: i8)
// CHECK-GENERIC: "hw.module.extern"() {sym_name = "extern", module_type = !hw.modty<input foo : i32, input hello : i16, output baz : i8>, parameters = []} : () -> ()
hw.module.extern @extern(in %foo: i32, in %bar "hello": i16, out baz: i8)


// Generic format input tests

// CHECK: hw.module @generic_basic(in %{{[^ ]*}} a: i1, out nameOfPortInSV: i1, out "": i1, in %{{[^ ]*}} customName: i1, in %{{[^ ]*}} "very custom name": i32) {
// CHECK-NEXT: hw.output %{{[^ ]*}}, %{{[^ ]*}} : i1, i1
// CHECK-NEXT: }
// CHECK-GENERIC: "hw.module"() ({
// CHECK-GENERIC-NEXT: ^{{[^(]*}}(%{{[^ ]*}} : i1, %{{[^ ]*}} : i1, %{{[^ ]*}} : i32):
// CHECK-GENERIC-NEXT:   "hw.output"(%{{[^ ]*}}, %{{[^ ]*}}) : (i1, i1) -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "generic_basic", module_type = !hw.modty<input a : i1, output nameOfPortInSV : i1, output "" : i1, input customName : i1, input "very custom name" : i32>, parameters = []} : () -> ()
"hw.module"() ({
  ^0(%a_foo : i1, %b : i1, %c: i32):
    "hw.output"(%a_foo, %b) : (i1, i1) -> ()
  }) {
    "sym_name" = "generic_basic",
    "module_type" = !hw.modty<input a : i1, output nameOfPortInSV : i1, output "" : i1, input customName : i1, input "very custom name" : i32>,
    "parameters" = []
  } : () -> ()

// CHECK: hw.module @generic_param<p1: i42, "p2 with wack name": i1>() {
// CHECK-NEXT: hw.output
// CHECK-NEXT: }
// CHECK-GENERIC: "hw.module"() ({
// CHECK-GENERIC-NEXT:   "hw.output"() : () -> ()
// CHECK-GENERIC-NEXT: }) {sym_name = "generic_param", module_type = !hw.modty<>, parameters = [#hw.param.decl<"p1": i42>, #hw.param.decl<"p2 with wack name": i1>]} : () -> ()
"hw.module"() ({
  "hw.output"() : () -> ()
}) {
  "sym_name" = "generic_param",
  "module_type" = !hw.modty<>, 
  "parameters" = [#hw.param.decl<"p1": i42>, #hw.param.decl<"p2 with wack name": i1>]
} : () -> ()
