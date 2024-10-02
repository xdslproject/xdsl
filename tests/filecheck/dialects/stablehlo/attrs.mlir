// RUN: XDSL_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    "test.op"() {"default" = #stablehlo<precision DEFAULT>, "high" = #stablehlo<precision HIGH>, "highest" = #stablehlo<precision HIGHEST>} : () -> ()
"test.op"() {
    default = #stablehlo<precision DEFAULT>,
    high = #stablehlo<precision HIGH>,
    highest = #stablehlo<precision HIGHEST>
} : () -> ()

// CHECK-NEXT:    %token = "test.op"() : () -> !stablehlo.token
%token = "test.op"() : () -> (!stablehlo.token)

// CHECK-NEXT:    "test.op"() {"dot" = #stablehlo.dot<
// CHECK-NEXT:      lhs_batching_dimensions = [0],
// CHECK-NEXT:      rhs_batching_dimensions = [1],
// CHECK-NEXT:      lhs_contracting_dimensions = [2],
// CHECK-NEXT:      rhs_contracting_dimensions = [3]
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [1],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [3]
    >
} : () -> ()

// CHECK-NEXT:  }
