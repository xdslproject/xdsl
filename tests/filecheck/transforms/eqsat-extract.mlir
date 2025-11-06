// RUN: xdsl-opt -p eqsat-extract %s | filecheck %s

// CHECK:         func.func @trivial_no_arithmetic(%a : index, %b : index) -> index {
// CHECK-NEXT:      func.return %a : index
// CHECK-NEXT:    }
func.func @trivial_no_arithmetic(%a : index, %b : index) -> index {
  %a_eq = eqsat.eclass %a {"min_cost_index" = #builtin.int<0>} : index
  func.return %a_eq : index
}

// CHECK:         func.func @trivial_no_extraction(%a : index, %b : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a : index
// CHECK-NEXT:      func.return %a_eq : index
// CHECK-NEXT:    }
func.func @trivial_no_extraction(%a : index, %b : index) -> index {
  %a_eq = eqsat.eclass %a : index
  func.return %a_eq : index
}

// CHECK:         func.func @trivial_arithmetic(%a : index, %b : index) -> index {
// CHECK-NEXT:      func.return %a : index
// CHECK-NEXT:    }
func.func @trivial_arithmetic(%a : index, %b : index) -> index {
  %one = arith.constant {"eqsat_cost" = #builtin.int<1>} 1 : index
  %one_eq = eqsat.eclass %one {"min_cost_index" = #builtin.int<0>} : index
  %amul = arith.muli %a_eq, %one_eq {"eqsat_cost" = #builtin.int<2>} : index
  %a_eq = eqsat.eclass %amul, %a {"min_cost_index" = #builtin.int<1>} : index
  func.return %a_eq : index
}

// CHECK:         func.func @non_trivial(%a : index, %b : index) -> index {
// CHECK-NEXT:      %two = arith.constant 2 : index
// CHECK-NEXT:      %a_times_two = arith.muli %a, %two : index
// CHECK-NEXT:      func.return %a_times_two : index
// CHECK-NEXT:    }
func.func @non_trivial(%a : index, %b : index) -> index {
  %a_eq = eqsat.eclass %a {"min_cost_index" = #builtin.int<0>} : index
  %one = arith.constant {"eqsat_cost" = #builtin.int<1000>} 1 : index
  %one_eq = eqsat.eclass %one {"min_cost_index" = #builtin.int<0>} : index
  %two = arith.constant {"eqsat_cost" = #builtin.int<1>} 2 : index
  %two_eq = eqsat.eclass %two {"min_cost_index" = #builtin.int<0>} : index
  %a_shift_one = arith.shli %a_eq, %one_eq {"eqsat_cost" = #builtin.int<1001>} : index
  %a_times_two = arith.muli %a_eq, %two_eq {"eqsat_cost" = #builtin.int<2>} : index
  %res_eq = eqsat.eclass %a_shift_one, %a_times_two {"min_cost_index" = #builtin.int<1>} : index
  func.return %res_eq : index
}

// CHECK:         func.func @partial_extraction(%a : index, %b : index) -> index {
// CHECK-NEXT:      %one = arith.constant 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one : index
// CHECK-NEXT:      %two = arith.constant 2 : index
// CHECK-NEXT:      %a_shift_one = arith.shli %a, %one_eq : index
// CHECK-NEXT:      %a_times_two = arith.muli %a, %two {eqsat_cost = #builtin.int<2>} : index
// CHECK-NEXT:      %res_eq = eqsat.eclass %a_shift_one, %a_times_two : index
// CHECK-NEXT:      func.return %res_eq : index
// CHECK-NEXT:    }
func.func @partial_extraction(%a : index, %b : index) -> index {
  %a_eq = eqsat.eclass %a {"min_cost_index" = #builtin.int<0>} : index
  %one = arith.constant 1 : index
  %one_eq = eqsat.eclass %one : index
  %two = arith.constant {"eqsat_cost" = #builtin.int<1>} 2 : index
  %two_eq = eqsat.eclass %two {"min_cost_index" = #builtin.int<0>} : index
  %a_shift_one = arith.shli %a_eq, %one_eq : index
  %a_times_two = arith.muli %a_eq, %two_eq {"eqsat_cost" = #builtin.int<2>} : index
  %res_eq = eqsat.eclass %a_shift_one, %a_times_two : index
  func.return %res_eq : index
}


// CHECK:         func.func @cycles(%a : i32) -> i32 {
// CHECK-NEXT:      func.return %a : i32
// CHECK-NEXT:    }
func.func @cycles(%a : i32) -> i32 {
  %two = arith.constant {eqsat_cost = #builtin.int<1>} 2 : i32
  %two_1 = eqsat.eclass %two {min_cost_index = #builtin.int<0>} : i32
  %mul = arith.muli %div, %two_1 {eqsat_cost = #builtin.int<1>} : i32
  %mul_1 = eqsat.eclass %mul {min_cost_index = #builtin.int<0>} : i32
  %0 = arith.constant {eqsat_cost = #builtin.int<1>} 1 : i32
  %1 = eqsat.const_eclass %0, %2 (constant = 1 : i32) {min_cost_index = #builtin.int<0>} : i32
  %2 = arith.divui %two_1, %two_1 {eqsat_cost = #builtin.int<1>} : i32
  %3 = arith.muli %div, %1 {eqsat_cost = #builtin.int<1>} : i32
  %div_1 = arith.divui %mul_1, %two_1 {eqsat_cost = #builtin.int<1>} : i32
  %div = eqsat.eclass %div_1, %3, %a {min_cost_index = #builtin.int<2>} : i32
  func.return %div : i32
}
