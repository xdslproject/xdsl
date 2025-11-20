// RUN: xdsl-opt -p eqsat-add-costs{default=1000} --verify-diagnostics --split-input-file %s | filecheck %s

//      CHECK:    func.func @recursive(%a : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a, %b {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %one = arith.constant {eqsat_cost = #builtin.int<1000>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one {min_cost_index = #builtin.int<0>} : index
// CHECK-NEXT:      %b = arith.muli %a_eq, %one_eq {eqsat_cost = #builtin.int<1000>} : index
// CHECK-NEXT:      func.return %a_eq : index
// CHECK-NEXT:    }

func.func @recursive(%a : index) -> (index) {
    %a_eq = eqsat.eclass %a, %b : index
    %one = arith.constant 1 : index
    %one_eq = eqsat.eclass %one : index
    %b = arith.muli %a_eq, %one_eq : index
    return %a_eq : index
}
