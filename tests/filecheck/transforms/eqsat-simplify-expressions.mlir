// RUN: xdsl-opt -p eqsat-add-costs %s | filecheck %s

func.func @test(%a : index, %b : index) -> (index) {
    %a_eq   = eqsat.eclass %a : index
    %one    = arith.constant 1 : index 
    %one_eq = eqsat.eclass %one : index
    %amul = arith.muli %a_eq, %one_eq   : index 
    
    %out  = eqsat.eclass %amul, %a_eq : index
    func.return %out : index
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @test(%a : index, %b : index) -> index {
// CHECK-NEXT:      %a_eq = eqsat.eclass %a {"cost" = #builtin.int<1>} : index
// CHECK-NEXT:      %one = arith.constant {"cost" = #builtin.int<1>} 1 : index
// CHECK-NEXT:      %one_eq = eqsat.eclass %one : index
// CHECK-NEXT:      %amul = arith.muli %a_eq, %one_eq : index
// CHECK-NEXT:      %out = eqsat.eclass %amul, %a_eq : index
// CHECK-NEXT:      func.return %out : index
// CHECK-NEXT:    }
// CHECK-NEXT:  }
