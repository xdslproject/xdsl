// RUN: XDSL_ROUNDTRIP

"test.op"() {
    default = #stablehlo<precision DEFAULT>,
    high = #stablehlo<precision HIGH>,
    highest = #stablehlo<precision HIGHEST>
} : () -> ()

%token = "test.op"() : () -> (!stablehlo.token)

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {"default" = #stablehlo<precision DEFAULT>, "high" = #stablehlo<precision HIGH>, "highest" = #stablehlo<precision HIGHEST>} : () -> ()
// CHECK-NEXT:    %token = "test.op"() : () -> !stablehlo.token
// CHECK-NEXT:  }
