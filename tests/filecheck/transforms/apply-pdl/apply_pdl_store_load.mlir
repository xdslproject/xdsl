// RUN: xdsl-opt %s -p apply-pdl | filecheck %s


// CHECK:         func.func @test(%x : memref<1xindex>) -> index {
// CHECK-NEXT:      %a = arith.constant 2 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      memref.store %a, %x[%c0] : memref<1xindex>
// CHECK-NEXT:      %d = arith.addi %a, %a : index
// CHECK-NEXT:      func.return %d : index
// CHECK-NEXT:    }

func.func @test(%x : memref<1xindex>) -> (index) {
  %a = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  memref.store %a, %x[%c0] : memref<1xindex>
  %b = memref.load %x[%c0] : memref<1xindex>
  %c = memref.load %x[%c0] : memref<1xindex>
  %d = arith.addi %b, %c : index
  func.return %d : index
}

pdl.pattern : benefit(1) {
  %memref_ty = pdl.type : memref<1xindex>
  %idx_ty = pdl.type : index

  %mem = pdl.operand : %memref_ty
  %idx = pdl.operand : %idx_ty
  %val = pdl.operand : %idx_ty

  %store = pdl.operation "memref.store" (%val, %mem, %idx : !pdl.value, !pdl.value, !pdl.value)
  %load = pdl.operation "memref.load" (%mem, %idx : !pdl.value, !pdl.value) -> (%idx_ty : !pdl.type)
  %loaded = pdl.result 0 of %load

  pdl.rewrite %load {
    pdl.replace %load with (%val : !pdl.value)
  }
}
