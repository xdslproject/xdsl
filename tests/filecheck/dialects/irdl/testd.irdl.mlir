// RUN:XDSL_ROUNDTRIP

builtin.module {
  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK:      irdl.type @parametric {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.parameters(%{{.*}})
    // CHECK-NEXT: }
    irdl.type @parametric {
      %0 = irdl.any
      irdl.parameters(%0)
    }

    // CHECK:      irdl.attribute @parametric_attr {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.parameters(%{{.*}})
    // CHECK-NEXT: }
    irdl.attribute @parametric_attr {
      %0 = irdl.any
      irdl.parameters(%0)
    }

    // CHECK:      irdl.type @attr_in_type_out {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.parameters(%{{.*}})
    // CHECK-NEXT: }
    irdl.type @attr_in_type_out {
      %0 = irdl.any
      irdl.parameters(%0)
    }

    // CHECK:      irdl.operation @eq {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @eq {
      %0 = irdl.is i32
      irdl.results(%0)
    }

    // CHECK:      irdl.operation @anyof {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @anyof {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(%2)
    }

    // CHECK:      irdl.operation @all_of {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   %{{.*}} = irdl.all_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @all_of {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.all_of(%2, %1)
      irdl.results(%3)
    }

    // CHECK:      irdl.operation @any {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(%0)
    }

    // CHECK:      irdl.operation @dynbase {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @dynbase {
      %0 = irdl.any
      %1 = irdl.parametric @parametric<%0>
      irdl.results(%1)
    }

    // CHECK:      irdl.operation @dynparams {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK-NEXT:   irdl.results(%{{.*}})
    // CHECK-NEXT: }
    irdl.operation @dynparams {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @parametric<%2>
      irdl.results(%3)
    }

    // CHECK:      irdl.operation @constraint_vars {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(%{{.*}}, %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(%2, %2)
    }
  }
}
