// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
  hw.module @test_empty() {
    hw.output
  }

  hw.module @test_basic(in %testIn: i32, out testOut: i32) {
    hw.output %testIn : i32
  }

  hw.module @test_multiple(in %testIn1: i32, in %testIn2: i1, out testOut1: i1, out testOut2: i32) {
    hw.output %testIn2, %testIn1 : i1, i32
  }

  %ci32 = "test.op"() : () -> (i32)
  %ci1 = "test.op"() : () -> (i1)

  // CHECK: hw.instance "empty_instance" @test_empty() -> ()
  // CHECK-GENERIC: "hw.instance"() {instanceName = "empty_instance", moduleName = @test_empty, argNames = [], resultNames = []} : () -> ()
  hw.instance "empty_instance" @test_empty() -> ()

  // CHECK: hw.instance "empty_instance_with_attrs" @test_empty() -> () {foo = 4 : i32, bar}
  // CHECK-GENERIC: "hw.instance"() {foo = 4 : i32, bar, instanceName = "empty_instance_with_attrs", moduleName = @test_empty, argNames = [], resultNames = []} : () -> ()
  hw.instance "empty_instance_with_attrs" @test_empty() -> () {foo = 4 : i32, bar}

  // CHECK: hw.instance "empty_instance_with_inner_sym" sym #hw<innerSym[]> @test_empty() -> ()
  // CHECK-GENERIC: "hw.instance"() {instanceName = "empty_instance_with_inner_sym", moduleName = @test_empty, argNames = [], resultNames = [], inner_sym = #hw<innerSym[]>} : () -> ()
  hw.instance "empty_instance_with_inner_sym" sym #hw<innerSym[]> @test_empty() -> ()

  // CHECK: %{{.*}} = hw.instance "basic_instance" @test_basic(testIn: %{{.*}}: i32) -> (testOut: i32)
  // CHECK-GENERIC: %{{.*}} = "hw.instance"(%{{.*}}) {instanceName = "basic_instance", moduleName = @test_basic, argNames = ["testIn"], resultNames = ["testOut"]} : (i32) -> i32
  %res = hw.instance "basic_instance" @test_basic(testIn: %ci32: i32) -> (testOut: i32)

  // CHECK: %{{.*}} = hw.instance "multiple_instance" @test_multiple(testIn1: %{{.*}}: i32, testIn2: %{{.*}}: i1) -> (testOut1: i1, testOut2: i32)
  // CHECK-GENERIC: %{{.*}} = "hw.instance"(%{{.*}}, %{{.*}}) {instanceName = "multiple_instance", moduleName = @test_multiple, argNames = ["testIn1", "testIn2"], resultNames = ["testOut1", "testOut2"]} : (i32, i1) -> (i1, i32)
  %res1, %res2 = hw.instance "multiple_instance" @test_multiple(testIn1: %ci32: i32, testIn2: %ci1: i1) -> (testOut1: i1, testOut2: i32)

  // CHECK: %{{.*}} = hw.instance "multiple_instance_swap" @test_multiple(testIn2: %{{.*}}: i1, testIn1: %{{.*}}: i32) -> (testOut2: i32, testOut1: i1)
  // CHECK-GENERIC: %{{.*}} = "hw.instance"(%{{.*}}, %{{.*}}) {instanceName = "multiple_instance_swap", moduleName = @test_multiple, argNames = ["testIn2", "testIn1"], resultNames = ["testOut2", "testOut1"]} : (i1, i32) -> (i32, i1)
  %res2s, %res1s = hw.instance "multiple_instance_swap" @test_multiple(testIn2: %ci1: i1, testIn1: %ci32: i32) -> (testOut2: i32, testOut1: i1)

  // Assert types.
  // CHECK: "test.op"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i1, i1, i32, i32) -> ()
  // CHECK-GENERIC: "test.op"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i1, i1, i32, i32) -> ()
  "test.op"(%res1, %res1s, %res2, %res2s) : (i1, i1, i32, i32) -> ()
}
