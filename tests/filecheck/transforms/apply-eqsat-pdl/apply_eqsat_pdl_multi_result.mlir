builtin.module {
  pdl.pattern : benefit(1) {
    %x = pdl.operand
    %y = pdl.operand
    %z = pdl.operand
    %type = pdl.type
    %accumop = pdl.operation "arith.addi" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %accum = pdl.result 0 of %accumop
    %resultop = pdl.operation "arith.subi" (%accum, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %result = pdl.result 0 of %resultop
    pdl.rewrite %resultop {
      %newaccumop = pdl.operation "arith.subi" (%y, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
      %newaccum = pdl.result 0 of %newaccumop
      %newresultop = pdl.operation "arith.addi" (%x, %newaccum : !pdl.value, !pdl.value) -> (%type : !pdl.type)
      %newresult = pdl.result 0 of %newresultop
      pdl.replace %resultop with %newresultop
    }
  }
  func.func @impl() -> i32 {
    %a = arith.constant 3 : i32
    %a_1 = eqsat.eclass %a : i32
    %b = arith.constant 5 : i32
    %b_1 = eqsat.eclass %b : i32
    %c = arith.constant 7 : i32
    %c_1 = eqsat.eclass %c : i32
    %d = arith.addi %a_1, %b_1 : i32
    %d_1 = eqsat.eclass %d : i32
    %e = arith.subi %d_1, %c_1 : i32
    %e_1 = eqsat.eclass %e : i32
    func.return %e_1 : i32
  }
}

// RUN: xdsl-opt %s -p apply-pdl

// pdl.pattern : benefit(1) {
//     %x = pdl.operand
//     %y = pdl.operand
//     %z = pdl.operand
//     %type = pdl.type
    
//     %accumop = pdl.operation "arith.addi" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//     %accum = pdl.result 0 of %accumop
    
//     %resultop = pdl.operation "arith.addi" (%accum, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//     %result = pdl.result 0 of %resultop
    
//     pdl.rewrite %resultop {
//         %newaccumop = pdl.operation "arith.addi" (%y, %z: !pdl.value, !pdl.value) -> (%type : !pdl.type)
//         %newaccum = pdl.result 0 of %newaccumop
//         %newresultop = pdl.operation "arith.addi" (%x, %newaccum: !pdl.value, !pdl.value) -> (%type : !pdl.type)
//         %newresult = pdl.result 0 of %newresultop
//         pdl.replace %accumop with %newaccumop
//         pdl.replace %resultop with %newresultop
//     }
// }

// func.func @impl() -> i32 {
//     %a = arith.constant 3 : i32
//     %b = arith.constant 5 : i32
//     %c = arith.constant 7 : i32
    
//     %d = arith.addi %a, %b : i32
//     %e = arith.addi %d, %c : i32
//     func.return %e : i32
// }
