// RUN: XDSL_ROUNDTRIP

// CHECK: module
"builtin.module"() ({
  // CHECK: func.func
  "func.func"() ({
    // CHECK-NEXT: test = #builtin.int<0>
    "test.op"() {test = #builtin.int<0>} : () -> ()
    // CHECK-NEXT: test = #builtin.int<5444517870735015415413993718908291383295>
    "test.op"() {test = #builtin.int<5444517870735015415413993718908291383295>} : () -> ()
    // CHECK-NEXT: test = #builtin.float_data<-0.4>
    "test.op"() {test = #builtin.float_data<-0.4>} : () -> ()
    // CHECK-NEXT: test = #builtin.float_data<2.1>
    "test.op"() {test = #builtin.float_data<2.100000e+00>} : () -> ()
    // CHECK-NEXT: test = true
    "test.op"() {test = true} : () -> ()
    // CHECK-NEXT: test = false
    "test.op"() {test = false} : () -> ()
    // CHECK-NEXT: test = 0 : ui1
    "test.op"() {test = 0 : ui1} : () -> ()
    // CHECK-NEXT: test = 1 : ui1
    "test.op"() {test = 1 : ui1} : () -> ()
    // CHECK-NEXT: test = 0 : si1
    "test.op"() {test = 0 : si1} : () -> ()
    // CHECK-NEXT: test = -1 : si1
    "test.op"() {test = -1 : si1} : () -> ()
    // CHECK-NEXT: test = array<i32: 2, 3, 4>
    "test.op"() {test = array<i32: 2, 3, 4>} : () -> ()
    // CHECK-NEXT: test = array<f32: 2.100000e+00, 3.200000e+00, 4.300000e+00>
    "test.op"() {test = array<f32: 2.1, 3.2, 4.3>} : () -> ()
    // CHECK-NEXT: test = #builtin.signedness<signless>
    "test.op"() {test = #builtin.signedness<signless>} : () -> ()
    // CHECK-NEXT: test = #builtin.signedness<signed>
    "test.op"() {test = #builtin.signedness<signed>} : () -> ()
    // CHECK-NEXT: test = #builtin.signedness<unsigned>
    "test.op"() {test = #builtin.signedness<unsigned>} : () -> ()
    // CHECK-NEXT: test = @foo
    "test.op"() {test = @foo} : () -> ()
    // CHECK-NEXT: test = @foo::@bar
    "test.op"() {test = @foo::@bar} : () -> ()
    // CHECK-NEXT: test = @foo::@bar::@baz
    "test.op"() {test = @foo::@bar::@baz} : () -> ()
    // CHECK-NEXT: test = loc(unknown)
    "test.op"() {test = loc(unknown)} : () -> ()
    // CHECK-NEXT: test = none
    "test.op"() {test = none} : () -> ()
    // CHECK-NEXT: test = 0 : i0
    "test.op"() {test = 0 : i0} : () -> ()
    // CHECK-NEXT: test = dense<0> : tensor<1xi32>
    "test.op"() {test = dense<0> : tensor<1xi32>} : () -> ()
    // CHECK-NEXT: test = dense<0.000000e+00> : tensor<1xf32>
    "test.op"() {test = dense<0.0> : tensor<1xf32>} : () -> ()
    // CHECK-NEXT: test = dense<255> : vector<ui8>
    "test.op"() {test = dense<255> : vector<ui8>} : () -> ()
    // CHECK-NEXT: test = dense<[[-51, 24, -4], [-97, -74, 73], [67, -124, -109]]> : tensor<3x3xi8>
    "test.op"() {test = dense<"0xCD18FC9FB649438493"> : tensor<3x3xi8>} : () -> ()
    // CHECK-NEXT: test = dense<true>
    "test.op"() {test = dense<true> : tensor<1xi1>} : () -> ()
    // CHECK-NEXT: test = dense<false>
    "test.op"() {test = dense<false> : tensor<1xi1>} : () -> ()
    // CHECK-NEXT: test = dense<(1.200000e+00,3.400000e+00)>
    "test.op"() {test = dense<(1.2,3.4)> : tensor<1xcomplex<f32>>} : () -> ()
    // CHECK-NEXT: test = dense<(1.200000e+00,3.400000e+00)>
    "test.op"() {test = dense<[(1.2,3.4)]> : tensor<1xcomplex<f32>>} : () -> ()
    // CHECK-NEXT: test = dense<(1,2)>
    "test.op"() {test = dense<(1,2)> : tensor<1xcomplex<i32>>} : () -> ()
    // CHECK-NEXT: test = dense<(1,2)>
    "test.op"() {test = dense<[(1,2)]> : tensor<1xcomplex<i32>>} : () -> ()
    // CHECK-NEXT: test = dense<(true,false)>
    "test.op"() {test = dense<[(true,false)]> : tensor<1xcomplex<i1>>} : () -> ()
    "func.return"() : () -> ()
  }) {"function_type" = () -> (), "sym_name" = "builtin"} : () -> ()
  "test.op"() {"value"= {"one"=1 : i64, "two"=2 : i64, "three"="three"}} : () -> ()
}) : () -> ()
