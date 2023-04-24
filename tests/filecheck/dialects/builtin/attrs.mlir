// RUN: xdsl-opt -t mlir %s | xdsl-opt -f mlir -t mlir | filecheck %s

// CHECK: module
"builtin.module"() ({
  "func.func"() ({
    // CHECK: "test" = dense<0> : tensor<1xi32>
    %x1 = "arith.constant"() {"value" = 0 : i64, "test" = dense<0> : tensor<1xi32>} : () -> i64
    // CHECK: "test" = dense<0.0> : tensor<1xf32>
    %x2 = "arith.constant"() {"value" = 0 : i64, "test" = dense<0.0> : tensor<1xf32>} : () -> i64
    // CHECK:  "test" = true
    %x3 = "arith.constant"() {"value" = 0 : i64, "test" = true} : () -> i64
    // CHECK:  "test" = false
    %x4 = "arith.constant"() {"value" = 0 : i64, "test" = false} : () -> i64
    // CHECK:  "test" = true
    %x5 = "arith.constant"() {"value" = 0 : i64, "test" = true} : () -> i64
    // CHECK:  "test" = false
    %x6 = "arith.constant"() {"value" = 0 : i64, "test" = false} : () -> i64
    // CHECK:  "test" = array<i32: 2, 3, 4>
    %x7 = "arith.constant"() {"value" = 0 : i64, "test" = array<i32: 2, 3, 4>} : () -> i64
    // CHECK:  "test" = array<f32: 2.1, 3.2, 4.3>
    %x8 = "arith.constant"() {"value" = 0 : i64, "test" = array<f32: 2.1, 3.2, 4.3>} : () -> i64
    // CHECK:  "test" = #signedness<signless>
    %x9 = "arith.constant"() {"value" = 0 : i64, "test" = #signedness<signless>} : () -> i64
    // CHECK:  "test" = #signedness<signed>
    %x10 = "arith.constant"() {"value" = 0 : i64, "test" = #signedness<signed>} : () -> i64
    // CHECK:  "test" = #signedness<unsigned>
    %x11 = "arith.constant"() {"value" = 0 : i64, "test" = #signedness<unsigned>} : () -> i64
    // CHECK:  "test" = @foo
    %x12 = "arith.constant"() {"value" = 0 : i64, "test" = @foo} : () -> i64
    // CHECK:  "test" = @foo::@bar
    %x13 = "arith.constant"() {"value" = 0 : i64, "test" = @foo::@bar} : () -> i64
    // CHECK:  "test" = @foo::@bar::@baz
    %x14 = "arith.constant"() {"value" = 0 : i64, "test" = @foo::@bar::@baz} : () -> i64
    "func.return"() : () -> ()
  }) {"function_type" = () -> (), "sym_name" = "builtin"} : () -> ()
  "func.func"() ({
  // CHECK: "one"=1 : i64, "two"=2 : i64, "three"="three"
  }) {"sym_name" = "abc", "function_type" = () -> (), "value" = {"one"=1 : i64, "two"=2 : i64, "three"="three"}} : () -> ()
}) : () -> ()
