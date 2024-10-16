// RUN: xdsl-opt -p eqsat-extract-expressions -p dce %s | filecheck %s

func.func @test(%a : index, %b : index) -> (index) {
    %a_eq = eqsat.eclass %a : index
    %b_eq = eqsat.eclass %b : index
    %c_ab = arith.addi %a_eq, %b_eq   : index 
    %c_ba = arith.addi %b_eq, %a_eq   : index 
    %c_eq = eqsat.eclass %c_ab, %c_ba : index
    func.return %c_eq : index
}