// RUN: XDSL_ROUNDTRIP

%lb, %ub, %step = "test.op"() : () -> (index, index, index)
scf.for %i0 = %lb to %ub step %step {
  scf.for %i1 = %lb to %ub step %step {
    scf.for %i2 = %lb to %ub step %step {
    }
  }
}

// CHECK:       %lb, %ub, %step = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:  scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
