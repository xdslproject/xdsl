// RUN: XDSL_ROUNDTRIP

// Types that have cyclic references.
builtin.module {
  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK:      irdl.type @self_referencing {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @self_referencing<%{{.*}}>
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @self_referencing {
      %0 = irdl.any
      %1 = irdl.parametric @self_referencing<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(param: %3)
    }


    // CHECK:      irdl.type @type1 {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @type2<%{{.*}}>
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @type1 {
      %0 = irdl.any
      %1 = irdl.parametric @type2<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(param: %3)
    }

    // CHECK:      irdl.type @type2 {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @type1<%{{.*}}>
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @type2 {
        %0 = irdl.any
        %1 = irdl.parametric @type1<%0>
        %2 = irdl.is i32
        %3 = irdl.any_of(%1, %2)
        irdl.parameters(param: %3)
    }
  }
}
