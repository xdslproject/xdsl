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

  %test = "test.op"() : () -> !hw.array<6xi7>
  // CHECK-NEXT: %{{.*}} = "test.op"() : () -> !hw.array<6xi7>

  %const = "test.op"() : () -> i19
  %array = hw.array_create %const, %const : i19 
  %array1 = hw.array_create %const : i19 

  // CHECK: %{{.*}} = hw.array_create %const, %const : i19 

  %index = "test.op"() : () -> i1
  %element = hw.array_get %array[%index] : !hw.array<2xi19>, i1

  // CHECK: %{{.*}} = hw.array_get %{{.*}}[%{{.*}}] : !hw.array<2xi19>, i1


}) : () -> ()
