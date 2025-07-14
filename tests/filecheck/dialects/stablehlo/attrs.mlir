// RUN: XDSL_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    "test.op"() {default = #stablehlo<precision DEFAULT>, high = #stablehlo<precision HIGH>, highest = #stablehlo<precision HIGHEST>} : () -> ()
"test.op"() {
    default = #stablehlo<precision DEFAULT>,
    high = #stablehlo<precision HIGH>,
    highest = #stablehlo<precision HIGHEST>
} : () -> ()

// CHECK-NEXT:    %token = "test.op"() : () -> !stablehlo.token
%token = "test.op"() : () -> (!stablehlo.token)

// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
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

// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
// CHECK-NEXT:      lhs_contracting_dimensions = [0],
// CHECK-NEXT:      rhs_contracting_dimensions = [1]
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_contracting_dimensions = [0],
        rhs_contracting_dimensions = [1]
    >
} : () -> ()

// CHECK-NEXT:    "test.op"() {
// CHECK-SAME:      eq = #stablehlo<comparison_direction EQ>,
// CHECK-SAME:      ne = #stablehlo<comparison_direction NE>,
// CHECK-SAME:      ge = #stablehlo<comparison_direction GE>,
// CHECK-SAME:      gt = #stablehlo<comparison_direction GT>,
// CHECK-SAME:      le = #stablehlo<comparison_direction LE>,
// CHECK-SAME:      lt = #stablehlo<comparison_direction LT>
// CHECK-SAME:     } : () -> ()
"test.op"() {
  eq = #stablehlo<comparison_direction EQ>,
  ne = #stablehlo<comparison_direction NE>,
  ge = #stablehlo<comparison_direction GE>,
  gt = #stablehlo<comparison_direction GT>,
  le = #stablehlo<comparison_direction LE>,
  lt = #stablehlo<comparison_direction LT>
} : () -> ()

// CHECK-NEXT:    "test.op"() {
// CHECK-SAME:      notype = #stablehlo<comparison_type NOTYPE>,
// CHECK-SAME:      float = #stablehlo<comparison_type FLOAT>,
// CHECK-SAME:      totalorder = #stablehlo<comparison_type TOTALORDER>,
// CHECK-SAME:      signed = #stablehlo<comparison_type SIGNED>,
// CHECK-SAME:      unsigned = #stablehlo<comparison_type UNSIGNED>
// CHECK-SAME:    } : () -> ()
"test.op"() {
  notype = #stablehlo<comparison_type NOTYPE>,
  float = #stablehlo<comparison_type FLOAT>,
  totalorder = #stablehlo<comparison_type TOTALORDER>,
  signed = #stablehlo<comparison_type SIGNED>,
  unsigned = #stablehlo<comparison_type UNSIGNED>
} : () -> ()

// CHECK-NEXT:  }
