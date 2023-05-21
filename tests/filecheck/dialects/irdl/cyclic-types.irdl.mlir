// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

// Types that have cyclic references.
builtin.module {
  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK:   irdl.type @self_referencing {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   %{{.*}} = irdl.parametric @self_referencing<%{{.*}}>
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.parameters(%{{.*}})
    // CHECK: }
    irdl.type @self_referencing {
      %0 = irdl.any
      %1 = irdl.parametric @self_referencing<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(%3)
    }


    // CHECK:   irdl.type @type1 {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   %{{.*}} = irdl.parametric @type2<%{{.*}}>
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.parameters(%{{.*}})
    irdl.type @type1 {
      %0 = irdl.any
      %1 = irdl.parametric @type2<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(%3)
    }

    // CHECK:   irdl.type @type2 {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   %{{.*}} = irdl.parametric @type1<%{{.*}}>
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.parameters(%{{.*}})
    irdl.type @type2 {
        %0 = irdl.any
        %1 = irdl.parametric @type1<%0>
        %2 = irdl.is i32
        %3 = irdl.any_of(%1, %2)
        irdl.parameters(%3)
    }
  }
}
