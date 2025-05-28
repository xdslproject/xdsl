// RUN: XDSL_ROUNDTRIP

builtin.module {
  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK: irdl.type @singleton
    irdl.type @singleton

    // CHECK:      irdl.type @parametrized {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.parameters(param1: %{{.*}}, param2: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @parametrized {
      %0 = irdl.any
      %1 = irdl.is i32
      %2 = irdl.is i64
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(param1: %0, param2: %3)
    }

    // CHECK:      irdl.operation @any {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(out: %0)
    }
  }
}
