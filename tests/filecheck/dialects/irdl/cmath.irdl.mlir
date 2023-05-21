// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

builtin.module {
  // CHECK-LABEL: irdl.dialect @cmath {
  irdl.dialect @cmath {

    // CHECK: irdl.type @complex {
    // CHECK:   %{{.*}} = irdl.is f32
    // CHECK:   %{{.*}} = irdl.is f64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.parameters(%{{.*}})
    // CHECK: }
    irdl.type @complex {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      irdl.parameters(%2)
    }

    // CHECK: irdl.operation @norm {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   %{{.*}} = irdl.parametric @complex<%{{.*}}>
    // CHECK:   irdl.operands(%{{.*}})
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @norm {
      %0 = irdl.any
      %1 = irdl.parametric @complex<%0>
      irdl.operands(%1)
      irdl.results(%0)
    }

    // CHECK: irdl.operation @mul {
    // CHECK:   %{{.*}} = irdl.is f32
    // CHECK:   %{{.*}} = irdl.is f64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   %{{.*}} = irdl.parametric @complex<%{{.*}}>
    // CHECK:   irdl.operands(%{{.*}}, %{{.*}})
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
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
