// RUN: XDSL_ROUNDTRIP

builtin.module {
  // CHECK: irdl.dialect @cmath {
  irdl.dialect @cmath {

    // CHECK: irdl.type @complex {
    // CHECK-NEXT:   %{{.*}} = irdl.is f32
    // CHECK-NEXT:   %{{.*}} = irdl.is f64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.parameters(%{{.*}})
    // CHECK-NEXT: }
    irdl.type @complex {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      irdl.parameters(%2)
    }

    // CHECK: irdl.operation @norm {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @complex<%{{.*}}>
    // CHECK-NEXT:   irdl.operands(%{{.*}})
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @norm {
      %0 = irdl.any
      %1 = irdl.parametric @complex<%0>
      irdl.operands(%1)
      irdl.results(%0)
    }

    // CHECK: irdl.operation @mul {
    // CHECK-NEXT:   %{{.*}} = irdl.is f32
    // CHECK-NEXT:   %{{.*}} = irdl.is f64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @complex<%{{.*}}>
    // CHECK-NEXT:   irdl.operands(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @mul {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @complex<%2>
      irdl.operands(%3, %3)
      irdl.results(%3)
    }

  }
}
