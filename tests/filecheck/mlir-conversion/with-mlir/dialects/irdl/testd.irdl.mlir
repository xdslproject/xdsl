// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK:      irdl.type @parametric {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @parametric {
      %0 = irdl.any
      irdl.parameters(param: %0)
    }

    // CHECK:      irdl.attribute @parametric_attr {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.attribute @parametric_attr {
      %0 = irdl.any
      irdl.parameters(param: %0)
    }

    // CHECK:      irdl.type @attr_in_type_out {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @attr_in_type_out {
      %0 = irdl.any
      irdl.parameters(param: %0)
    }

    // CHECK:      irdl.operation @eq {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   irdl.results(res: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @eq {
      %0 = irdl.is i32
      irdl.results(res: %0)
    }

    // CHECK:      irdl.operation @any {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.results(res: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(res: %0)
    }

    // CHECK:      irdl.operation @dynbase {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @testd::@parametric<%{{.*}}>
    // CHECK-NEXT:   irdl.results(res: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @dynbase {
      %0 = irdl.any
      %1 = irdl.parametric @testd::@parametric<%0>
      irdl.results(res: %1)
    }

    // CHECK:      irdl.operation @dynparams {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @testd::@parametric<%{{.*}}>
    // CHECK-NEXT:   irdl.results(res: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @dynparams {
      %0 = irdl.is i32
      %3 = irdl.parametric @testd::@parametric<%0>
      irdl.results(res: %3)
    }

    // CHECK:      irdl.operation @constraint_vars {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   irdl.results(res0: %{{.*}}, res1: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      irdl.results(res0: %0, res1: %0)
    }
  }
