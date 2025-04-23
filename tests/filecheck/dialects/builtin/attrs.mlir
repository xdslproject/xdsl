// RUN: XDSL_ROUNDTRIP

// CHECK: module
"builtin.module"() ({
  "func.func"() ({
    // CHECK: test = dense<0> : tensor<1xi32>
    %x1 = "arith.constant"() {"value" = 0 : i64, "test" = dense<0> : tensor<1xi32>} : () -> i64
    // CHECK: test = dense<0.000000e+00> : tensor<1xf32>
    %x2 = "arith.constant"() {"value" = 0 : i64, "test" = dense<0.0> : tensor<1xf32>} : () -> i64
    // CHECK:  test = true
    %x3 = "arith.constant"() {"value" = 0 : i64, "test" = true} : () -> i64
    // CHECK:  test = false
    %x4 = "arith.constant"() {"value" = 0 : i64, "test" = false} : () -> i64
    // CHECK:  test = true
    %x5 = "arith.constant"() {"value" = 0 : i64, "test" = true} : () -> i64
    // CHECK:  test = false
    %x6 = "arith.constant"() {"value" = 0 : i64, "test" = false} : () -> i64
    // CHECK:  test = array<i32: 2, 3, 4>
    %x7 = "arith.constant"() {"value" = 0 : i64, "test" = array<i32: 2, 3, 4>} : () -> i64
    // CHECK:  test = array<f32: 2.0999999046325684, 3.200000047683716, 4.300000190734863>
    %x8 = "arith.constant"() {"value" = 0 : i64, "test" = array<f32: 2.1, 3.2, 4.3>} : () -> i64
    // CHECK:  test = #builtin.signedness<signless>
    %x9 = "arith.constant"() {"value" = 0 : i64, "test" = #builtin.signedness<signless>} : () -> i64
    // CHECK:  test = #builtin.signedness<signed>
    %x10 = "arith.constant"() {"value" = 0 : i64, "test" = #builtin.signedness<signed>} : () -> i64
    // CHECK:  test = #builtin.signedness<unsigned>
    %x11 = "arith.constant"() {"value" = 0 : i64, "test" = #builtin.signedness<unsigned>} : () -> i64
    // CHECK:  test = @foo
    %x12 = "arith.constant"() {"value" = 0 : i64, "test" = @foo} : () -> i64
    // CHECK:  test = @foo::@bar
    %x13 = "arith.constant"() {"value" = 0 : i64, "test" = @foo::@bar} : () -> i64
    // CHECK:  test = @foo::@bar::@baz
    %x14 = "arith.constant"() {"value" = 0 : i64, "test" = @foo::@bar::@baz} : () -> i64
    // CHECK:  test = loc(unknown)
    %x15 = "arith.constant"() {"value" = 0 : i64, "test" = loc(unknown)} : () -> i64
    // CHECK:  test = none
    %x16 = "arith.constant"() {"value" = 0 : i64, "test" = none} : () -> i64
    // CHECK:  test = 0 : i0
    %x17 = "arith.constant"() {"value" = 0 : i64, "test" = 0 : i0} : () -> i64
    // CHECK:  %x18 = arith.constant {array = array<i8: -1>, dense = dense<-1> : vector<i8>, test = -1 : i8} -1 : i8
    %x18 = arith.constant {"array" = array<i8: 255>, "dense" = dense<255> : vector<i8>, "test" = 255 : i8} -1 : i8
    // CHECK: %x19 = arith.constant dense<[[-51, 24, -4], [-97, -74, 73], [67, -124, -109]]> : tensor<3x3xi8>
    %x19 = arith.constant dense<"0xCD18FC9FB649438493"> : tensor<3x3xi8>
    "func.return"() : () -> ()
  }) {"function_type" = () -> (), "sym_name" = "builtin"} : () -> ()
  "test.op"() {"value"= {"one"=1 : i64, "two"=2 : i64, "three"="three"}} : () -> ()
}) : () -> ()
