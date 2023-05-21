// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

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

    // CHECK: irdl.attribute @parametric_attr {
    // CHECK:   %{{.*}} = irdl.any
    // CHECK:   irdl.parameters(%{{.*}})
    // CHECK: }
    irdl.attribute @parametric_attr {
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
    // CHECK:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK:   irdl.results(%{{.*}})
    // CHECK: }
    irdl.operation @dynparams {
      %0 = irdl.is i32
      %3 = irdl.parametric @parametric<%0>
      irdl.results(%3)
    }

    // CHECK: irdl.operation @constraint_vars {
    // CHECK:   %{{.*}} = irdl.is i32
    // CHECK:   irdl.results(%{{.*}}, %{{.*}})
    // CHECK: }
    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      irdl.results(%0, %0)
    }
  }
}
