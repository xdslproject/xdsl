// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
  "test.op"() { "test" = #hw.innerNameRef<@Foo::@Bar> } : () -> ()
  // CHECK:  "test.op"() {test = #hw.innerNameRef<@Foo::@Bar>} : () -> ()

  "test.op"() { "test" = #hw<innerSym[<@x_1,4,public>, <@y,5,public>]> } : () -> ()
  // CHECK-NEXT:  "test.op"() {test = #hw<innerSym[<@x_1,4,public>, <@y,5,public>]>} : () -> ()

  "test.op"() { "test" = #hw<innerSym@sym> } : () -> ()
  // CHECK-NEXT:  "test.op"() {test = #hw<innerSym@sym>} : () -> ()

  "test.op"() { "test" = #hw.direction<input> } : () -> ()
  // CHECK-NEXT:  "test.op"() {test = #hw.direction<input>} : () -> ()

  %test = "test.op"() : () -> !hw.array<6x9xi7>
  // CHECK-NEXT: %{{.*}} = "test.op"() : () -> !hw.array<6x9xi7>
}) : () -> ()
