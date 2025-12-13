// RUN: MLIR_ROUNDTRIP  
// RUN: MLIR_GENERIC_ROUNDTRIP  


 // CHECK: builtin.module {
builtin.module {

 // CHECK-NEXT:   func.func @execute_region() {
 // CHECK-NEXT:     %0 = scf.execute_region -> (i32) {
 // CHECK-NEXT:       %1 = "test.op"() : () -> i32
 // CHECK-NEXT:       scf.yield %1 : i32
 // CHECK-NEXT:     }
 // CHECK-NEXT:     func.return
 // CHECK-NEXT:   }

  func.func @execute_region() {
    %b = scf.execute_region -> (i32) {
      %a = "test.op"() : () -> (i32)
      scf.yield %a : i32
    }
    func.return
  }
  
  // CHECK:      func.func @execute_region_multiple_blocks() {
  // CHECK-NEXT:   %0 = scf.execute_region -> (i32) {
  // CHECK-NEXT:     %1 = "test.op"() : () -> i1
  // CHECK-NEXT:     cf.cond_br %1, ^bb0, ^bb1
  // CHECK-NEXT:   ^bb0:
  // CHECK-NEXT:     %2 = "test.op"() : () -> i32
  // CHECK-NEXT:     scf.yield %2 : i32
  // CHECK-NEXT:   ^bb1:
  // CHECK-NEXT:     %3 = "test.op"() : () -> i32
  // CHECK-NEXT:     scf.yield %3 : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }
  
  func.func @execute_region_multiple_blocks() {
    %c = scf.execute_region -> (i32) {
      %cond = "test.op"() : () -> i1
      cf.cond_br %cond, ^bb0, ^bb1
    ^bb0:
      %a = "test.op"() : () -> i32
      scf.yield %a : i32
    ^bb1:
      %b = "test.op"() : () -> i32
      scf.yield %b : i32
    }
    func.return
  }

 // CHECK-NEXT: }
}
