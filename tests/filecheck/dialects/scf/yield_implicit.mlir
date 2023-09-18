// RUN: XDSL_ROUNDTRIP

func.func @std_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = arith.cmpi slt, %i0, %i1 : index
      %min = arith.select %min_cmp, %i0, %i1 : index
      %max_cmp = arith.cmpi sge, %i0, %i1 : index
      %max = arith.select %max_cmp, %i0, %i1 : index
      scf.for %i2 = %min to %max step %i1 {
      }
    }
  }
  func.return
}

//  CHECK:      func.func @std_for(
//  CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = arith.cmpi slt, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = arith.cmpi sge, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

