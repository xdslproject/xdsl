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
