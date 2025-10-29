// RUN: xdsl-opt -p eqsat-add-costs{default=1} --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK:         func.func @trivial_arithmetic(%a : index, %b : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %one = arith.constant {eqsat_cost = #builtin.int<1>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %two = arith.constant {eqsat_cost = #builtin.int<1>} 2 : index
// CHECK-NEXT:      %two_eq = eqsat.eclass %two {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %a_shift_one = arith.shli %a_eq, %one_eq {eqsat_cost = #builtin.int<1>} : index
// CHECK-NEXT:      %a_times_two = arith.muli %a_eq, %two_eq {eqsat_cost = #builtin.int<1>} : index
// CHECK-NEXT:      %res_eq = eqsat.eclass %a_shift_one, %a_times_two {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      func.return %res_eq : index
// CHECK-NEXT:    }
func.func @trivial_arithmetic(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a : index
    %one = arith.constant 1 : index
    %one_eq = eqsat.eclass %one : index
    %two = arith.constant 2 : index
    %two_eq = eqsat.eclass %two : index
    %a_shift_one = arith.shli %a_eq, %one_eq : index
    %a_times_two = arith.muli %a_eq, %two_eq : index
    %res_eq = eqsat.eclass %a_shift_one, %a_times_two : index
    func.return %res_eq : index
}

// CHECK-NEXT:    func.func @no_eclass(%a : index, %b : index) -> index {
// CHECK-NEXT:      %one = arith.constant 1 : index
// CHECK-NEXT:      %amul = arith.muli %a, %one : index
// CHECK-NEXT:      func.return %amul : index
// CHECK-NEXT:    }
func.func @no_eclass(%a : index, %b : index) -> (index) {
    %one = arith.constant 1 : index
    %amul = arith.muli %a, %one : index
    func.return %amul : index
}

// CHECK-NEXT:    func.func @existing_cost(%a : index, %b : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %one = arith.constant {eqsat_cost = #builtin.int<1000>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %two = arith.constant {eqsat_cost = #builtin.int<1>} 2 : index
// CHECK-NEXT:      %two_eq = eqsat.eclass %two {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %a_shift_one = arith.shli %a_eq, %one_eq {eqsat_cost = #builtin.int<1>} : index
// CHECK-NEXT:      %a_times_two = arith.muli %a_eq, %two_eq {eqsat_cost = #builtin.int<1>} : index
// CHECK-NEXT:      %res_eq = eqsat.eclass %a_shift_one, %a_times_two {min_cost_index = #builtin.int<1>} : index
// CHECK-NEXT:      func.return %res_eq : index
// CHECK-NEXT:    }
func.func @existing_cost(%a : index, %b : index) -> (index) {
    // Another pass can set the cost, which must not be overwritten
    %a_eq = eqsat.eclass %a : index
    %one = arith.constant {"eqsat_cost" = #builtin.int<1000>} 1  : index
    %one_eq = eqsat.eclass %one : index
    %two = arith.constant 2 : index
    %two_eq = eqsat.eclass %two : index
    %a_shift_one = arith.shli %a_eq, %one_eq : index
    %a_times_two = arith.muli %a_eq, %two_eq : index
    %res_eq = eqsat.eclass %a_shift_one, %a_times_two : index
    func.return %res_eq : index
}

// -----

//      CHECK:    func.func @recursive(%a : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a, %b {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %one = arith.constant {eqsat_cost = #builtin.int<1>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %b = arith.muli %a_eq, %one_eq {eqsat_cost = #builtin.int<1>} : index
// CHECK-NEXT:      func.return %a_eq : index
// CHECK-NEXT:    }

func.func @recursive(%a : index) -> (index) {
    %a_eq = eqsat.eclass %a, %b : index
    %one = arith.constant 1 : index
    %one_eq = eqsat.eclass %one : index
    %b = arith.muli %a_eq, %one_eq : index
    return %a_eq : index
}

// -----

// CHECK:    Unexpected value 1000 : i64 for key eqsat_cost in ConstantOp(%one = arith.constant {eqsat_cost = 1000 : i64} 1 : index)

func.func @wrong_type_cost(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a : index
    %one = arith.constant {"eqsat_cost" = 1000} 1  : index
    %one_eq = eqsat.eclass %one : index
    %two = arith.constant 2 : index
    %two_eq = eqsat.eclass %two : index
    %a_times_two = arith.muli %a_eq, %two_eq : index
    %a_shift_one = arith.shli %a_eq, %one_eq : index
    %res_eq = eqsat.eclass %a_shift_one, %a_times_two : index
    func.return %res_eq : index
}

// -----

// CHECK:  Cannot compute cost of one result of operation with multiple results: TestOp(%test0, %test1 = "test.op"() : () -> (index, index))

func.func @multiple_results(%a : index, %b : index) -> (index) {
    %test0, %test1 = "test.op"() : () -> (index, index)
    %out = eqsat.eclass %test0, %test1 : index
    func.return %out : index
}
