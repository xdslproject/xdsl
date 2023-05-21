// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

builtin.module {
  // CHECK-LABEL: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK: irdl.type @singleton
    irdl.type @singleton

    // CHECK: irdl.type @parametrized {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.is i64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.parameters(%{{.*}}, %{{.*}})
    // CHECK: }
    irdl.type @parametrized {
      %0 = irdl.any
      %1 = irdl.is i32
      %2 = irdl.is i64
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(%0, %3)
    }

    // CHECK: irdl.operation @any {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(%0)
    }
  }
}
