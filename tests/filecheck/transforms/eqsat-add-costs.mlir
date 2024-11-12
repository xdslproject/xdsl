// RUN: xdsl-opt -p eqsat-add-costs --verify-diagnostics --split-input-file %s | filecheck %s

// CHECK:         func.func @trivial_arithmetic(%a : index, %b : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a {"eqsat_cost" = #builtin.int<0>} : index
// CHECK-NEXT:      %one = arith.constant {"eqsat_cost" = #builtin.int<1>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one {"eqsat_cost" = #builtin.int<1>} : index
// CHECK-NEXT:      %amul = arith.muli %a_eq, %one_eq {"eqsat_cost" = #builtin.int<2>} : index
// CHECK-NEXT:      %out = eqsat.eclass %amul, %a_eq {"eqsat_cost" = #builtin.int<0>} : index
// CHECK-NEXT:      func.return %out : index
// CHECK-NEXT:    }
func.func @trivial_arithmetic(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a : index
    %one = arith.constant 1 : index
    %one_eq = eqsat.eclass %one : index
    %amul = arith.muli %a_eq, %one_eq : index
    %out = eqsat.eclass %amul, %a_eq : index
    func.return %out : index
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
// CHECK-NEXT:      %a_eq = eqsat.eclass %a {"eqsat_cost" = #builtin.int<0>} : index
// CHECK-NEXT:      %one = arith.constant {"eqsat_cost" = #builtin.int<1000>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one {"eqsat_cost" = #builtin.int<1000>} : index
// CHECK-NEXT:      %amul = arith.muli %a_eq, %one_eq {"eqsat_cost" = #builtin.int<1001>} : index
// CHECK-NEXT:      %out = eqsat.eclass %amul, %a_eq {"eqsat_cost" = #builtin.int<0>} : index
// CHECK-NEXT:      func.return %out : index
// CHECK-NEXT:    }
func.func @existing_cost(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a : index
    // Another pass can set the cost, which must not be overwritten
    %one = arith.constant {"eqsat_cost" = #builtin.int<1000>} 1  : index
    %one_eq = eqsat.eclass %one : index
    %amul = arith.muli %a_eq, %one_eq : index
    %out = eqsat.eclass %amul, %a_eq : index
    func.return %out : index
}

// -----

// CHECK:    Unexpected value 1000 : i64 for key eqsat_cost in Constant(%one = arith.constant {"eqsat_cost" = 1000 : i64} 1 : index)

func.func @wrong_type_cost(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a  : index
    %one = arith.constant {"eqsat_cost" = 1000} 1 : index
    %one_eq = eqsat.eclass %one : index
    %amul = arith.muli %a_eq, %one_eq : index
    %out = eqsat.eclass %amul, %a_eq : index
    func.return %out : index
}

// -----

// CHECK:  Cannot compute cost of one result of operation with multiple results: TestOp(%test0, %test1 = "test.op"() : () -> (index, index))

func.func @multiple_results(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a  : index
    %test0, %test1 = "test.op"() : () -> (index, index)
    %out = eqsat.eclass %test0, %a_eq : index
    func.return %out : index
}
