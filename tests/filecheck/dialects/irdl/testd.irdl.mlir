// RUN:XDSL_ROUNDTRIP

builtin.module {
  // CHECK: irdl.dialect @testd {
  irdl.dialect @testd {

    // CHECK:      irdl.type @base_name {
    // CHECK-NEXT:   %{{.*}} = irdl.base "!my_dialect.type_name"
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @base_name {
      %0 = irdl.base "!my_dialect.type_name"
      irdl.parameters(param: %0)
    }

    // CHECK:      irdl.type @base_ref {
    // CHECK-NEXT:   %{{.*}} = irdl.base @base_name
    // CHECK-NEXT:   irdl.parameters(param: %{{.*}})
    // CHECK-NEXT: }
    irdl.type @base_ref {
      %0 = irdl.base @base_name
      irdl.parameters(param: %0)
    }

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
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @eq {
      %0 = irdl.is i32
      irdl.results(out: %0)
    }

    // CHECK:      irdl.operation @anyof {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @anyof {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(out: %2)
    }

    // CHECK:      irdl.operation @all_of {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   %{{.*}} = irdl.all_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @all_of {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.all_of(%2, %1)
      irdl.results(out: %3)
    }

    // CHECK:      irdl.operation @any {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(out: %0)
    }

    // CHECK:      irdl.operation @dynbase {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @dynbase {
      %0 = irdl.any
      %1 = irdl.parametric @parametric<%0>
      irdl.results(out: %1)
    }

    // CHECK:      irdl.operation @dynparams {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   %{{.*}} = irdl.parametric @parametric<%{{.*}}>
    // CHECK-NEXT:   irdl.results(out: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @dynparams {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @parametric<%2>
      irdl.results(out: %3)
    }

    // CHECK:      irdl.operation @constraint_vars {
    // CHECK-NEXT:   %{{.*}} = irdl.is i32
    // CHECK-NEXT:   %{{.*}} = irdl.is i64
    // CHECK-NEXT:   %{{.*}} = irdl.any_of(%{{.*}}, %{{.*}})
    // CHECK-NEXT:   irdl.results(out1: %{{.*}}, out2: %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(out1: %2, out2: %2)
    }

    // CHECK:      irdl.operation @variadicity {
    // CHECK-NEXT:   %{{.*}} = irdl.any
    // CHECK-NEXT:   irdl.operands(normal: %{{.*}}, sin: %{{.*}}, opt: optional %{{.*}}, var: variadic %{{.*}})
    // CHECK-NEXT:   irdl.results(normal: %{{.*}}, sin: %{{.*}}, opt: optional %{{.*}}, var: variadic %{{.*}})
    // CHECK-NEXT: }
    irdl.operation @variadicity {
      %0 = irdl.any
      irdl.operands(normal: %0, sin: single %0, opt: optional %0, var: variadic %0)
      irdl.results(normal: %0, sin: single %0, opt: optional %0, var: variadic %0)
    }

    // CHECK:      irdl.operation @op_with_regions {
    // CHECK-NEXT:    %r0 = irdl.region
    // CHECK-NEXT:    %r1 = irdl.region()
    // CHECK-NEXT:    %v0 = irdl.is i32
    // CHECK-NEXT:    %v1 = irdl.is i64
    // CHECK-NEXT:    %r2 = irdl.region(%v0, %v1)
    // CHECK-NEXT:    %r3 = irdl.region with size 3
    // CHECK-NEXT:    irdl.regions(r0: %r0, r1: %r1, r2: %r2, r3: %r3)
    // CHECK-NEXT:  }
    irdl.operation @op_with_regions {
      %r0 = irdl.region
      %r1 = irdl.region()
      %v0 = irdl.is i32
      %v1 = irdl.is i64
      %r2 = irdl.region(%v0, %v1)
      %r3 = irdl.region with size 3

      irdl.regions(r0: %r0, r1: %r1, r2: %r2, r3: %r3)
    }

    // CHECK:      irdl.operation @attr_op {
    // CHECK-NEXT:   %[[#first:]] = irdl.any
    // CHECK-NEXT:   %[[#second:]] = irdl.is i64
    // CHECK-NEXT:   irdl.attributes {
    // CHECK-NEXT:     "attr1" = %[[#first]],
    // CHECK-NEXT:     "attr2" = %[[#second]]
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    irdl.operation @attr_op {
      %0 = irdl.any
      %1 = irdl.is i64
      irdl.attributes {
        "attr1" = %0,
        "attr2" = %1
      }
    }

    // CHECK:      irdl.operation @no_attrs {
    // CHECK-NEXT:   irdl.attributes
    // CHECK-NEXT: }
    irdl.operation @no_attrs {
      irdl.attributes
    }
  }
}
