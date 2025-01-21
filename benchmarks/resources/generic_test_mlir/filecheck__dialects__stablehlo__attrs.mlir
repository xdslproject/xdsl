"builtin.module"() ({
  "test.op"() {default = #stablehlo<precision DEFAULT>, high = #stablehlo<precision HIGH>, highest = #stablehlo<precision HIGHEST>} : () -> ()
  %0 = "test.op"() : () -> !stablehlo.token
  "test.op"() {dot = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [1],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [3]
    >} : () -> ()
}) : () -> ()
