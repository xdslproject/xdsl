// RUN: xdsl-opt -p 'eqsat-add-costs{cost_file="%p/costs.json"}' --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK:      func.func @trivial_arithmetic(%a : i32, %b : i32) -> i32 {
// CHECK-NEXT: %a_eq = equivalence.class %a {min_cost_index = #builtin.int<0>} : i32
// CHECK-NEXT: %one = arith.constant {eqsat_cost = #builtin.int<1>} 1 : i32
// CHECK-NEXT: %one_eq = equivalence.class %one {min_cost_index = #builtin.int<0>} : i32
// CHECK-NEXT: %two = arith.constant {eqsat_cost = #builtin.int<1>} 2 : i32
// CHECK-NEXT: %two_eq = equivalence.class %two {min_cost_index = #builtin.int<0>} : i32
// CHECK-NEXT: %a_shift_one = arith.shli %a_eq, %one_eq {eqsat_cost = #builtin.int<2>} : i32
// CHECK-NEXT: %a_times_two = arith.muli %a_eq, %two_eq {eqsat_cost = #builtin.int<5>} : i32
// CHECK-NEXT: %res_eq = equivalence.class %a_shift_one, %a_times_two {min_cost_index = #builtin.int<0>} : i32
// CHECK-NEXT: func.return %res_eq : i32
// CHECK-NEXT: }

func.func @trivial_arithmetic(%a : i32, %b : i32) -> (i32) {
    %a_eq = equivalence.class %a : i32
    %one = arith.constant 1 : i32
    %one_eq = equivalence.class %one : i32
    %two = arith.constant 2 : i32
    %two_eq = equivalence.class %two : i32
    %a_shift_one = arith.shli %a_eq, %one_eq : i32
    %a_times_two = arith.muli %a_eq, %two_eq : i32
    %res_eq = equivalence.class %a_shift_one, %a_times_two : i32
    func.return %res_eq : i32
}
