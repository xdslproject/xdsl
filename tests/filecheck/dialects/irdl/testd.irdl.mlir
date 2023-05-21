// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

builtin.module {
  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK: irdl.type @parametric {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   irdl.parameters(%{{.*}})
    // CHECK: }
    irdl.type @parametric {
      %0 = irdl.any
      irdl.parameters(%0)
    }

    // CHECK: irdl.type @attr_in_type_out {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   irdl.parameters(%{{.*}})
    // CHECK: }
    irdl.type @attr_in_type_out {
      %0 = irdl.any
      irdl.parameters(%0)
    }

    // CHECK: irdl.operation @eq {
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @eq {
      %0 = irdl.is i32
      irdl.results(%0)
    }

    // CHECK: irdl.operation @anyof {
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.is i64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @anyof {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(%2)
    }

    // CHECK: irdl.operation @all_of {
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.is i64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   %{{.*}} = irdl.all_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @all_of {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.all_of(%2, %1)
      irdl.results(%3)
    }

    // CHECK: irdl.operation @any {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(%0)
    }

    // CHECK: irdl.operation @dynbase {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @dynbase {
      %0 = irdl.any
      %1 = irdl.parametric @parametric<%0>
      irdl.results(%1)
    }

    // CHECK: irdl.operation @dynparams {
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.is i64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @dynparams {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @parametric<%2>
      irdl.results(%3)
    }

    // CHECK: irdl.operation @constraint_vars {
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   %{{.*}} = irdl.is i64
    // CHECK:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK:   irdl.results(%{{.*}}, %{{.*}})
    // CHECK: }
    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(%2, %2)
    }
  }
}
