// RUN: xdsl-opt -p eqsat-extract-expressions %s | filecheck %s

func.func @test(%a : index, %b : index) -> (index) {
    %a_eq   = eqsat.eclass %a : index
    %one    = arith.constant 1 : index 
    %one_eq = eqsat.eclass %one : index
    %amul = arith.muli %a_eq, %one_eq   : index 
    
    %out  = eqsat.eclass %amul, %a_eq : index
    func.return %out : index
}
