// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s


builtin.module {

  func.func @execute_region() {
    %d = scf.execute_region -> (i32) {
      %a = arith.constant 0 : i32
      %b = arith.constant 1 : i32
      %c = arith.addi %a, %b : i32
      scf.yield %c : i32
    }
    func.return
  }
  
  // CHECK:      "func.func"() <{function_type = () -> (), sym_name = "execute_region"}> ({
  // CHECK-NEXT:   %0 = "scf.execute_region"() ({
  // CHECK-NEXT:     %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  // CHECK-NEXT:     %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  // CHECK-NEXT:     %3 = "arith.addi"(%1, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  // CHECK-NEXT:     "scf.yield"(%3) : (i32) -> ()
  // CHECK-NEXT:   }) : () -> i32
  // CHECK-NEXT:   "func.return"() : () -> ()
  // CHECK-NEXT: }) : () -> ()

}
