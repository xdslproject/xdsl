// RUN: xdsl-opt -p arith-add-fastmath %s | filecheck %s
// RUN: xdsl-opt -p "arith-add-fastmath{flags=none}" %s | filecheck %s --check-prefix=NONE
// RUN: xdsl-opt -p "arith-add-fastmath{flags=nnan}" %s | filecheck %s --check-prefix=SINGLE
// RUN: xdsl-opt -p "arith-add-fastmath{flags=nnan,nsz}" %s | filecheck %s --check-prefix=DOUBLE

func.func public @foo1() -> (f64, f64, f64, f64, f64, f64, i1) {
  %lhs, %rhs = "test.op"() : () -> (f64, f64)
  %4 = arith.addf %lhs, %rhs : f64
  %5 = arith.subf %lhs, %rhs : f64
  %6 = arith.mulf %lhs, %rhs : f64
  %7 = arith.divf %lhs, %rhs : f64
  %8 = arith.minimumf %lhs, %rhs : f64
  %9 = arith.maximumf %lhs, %rhs : f64
  %10 = arith.cmpf ole, %lhs, %rhs : f64
  return %4, %5, %6, %7, %8, %9, %10 : f64, f64, f64, f64, f64, f64, i1
}

// Check the default transform
// CHECK:       func.func public @foo1() -> (f64, f64, f64, f64, f64, f64, i1) {
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    func.return {{.*}}
// CHECK-NEXT:  }

// Check the transform with "none"
// NONE:       func.func public @foo1() -> (f64, f64, f64, f64, f64, f64, i1) {
// NONE-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// NONE-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    func.return {{.*}}
// NONE-NEXT:  }

// Check the transform with single flag
// SINGLE:       func.func public @foo1() -> (f64, f64, f64, f64, f64, f64, i1) {
// SINGLE-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// SINGLE-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    func.return {{.*}}
// SINGLE-NEXT:  }

// Check the transform with two flags
// DOUBLE:       func.func public @foo1() -> (f64, f64, f64, f64, f64, f64, i1) {
// DOUBLE-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// DOUBLE-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    func.return {{.*}}
// DOUBLE-NEXT:  }

func.func public @foo2() -> (f64, f64, f64, f64, f64, f64, i1) {
  %lhs, %rhs = "test.op"() : () -> (f64, f64)
  %4 = arith.addf %lhs, %rhs fastmath<fast> : f64
  %5 = arith.subf %lhs, %rhs fastmath<fast> : f64
  %6 = arith.mulf %lhs, %rhs fastmath<fast> : f64
  %7 = arith.divf %lhs, %rhs fastmath<fast> : f64
  %8 = arith.minimumf %lhs, %rhs fastmath<fast> : f64
  %9 = arith.maximumf %lhs, %rhs fastmath<fast> : f64
  %10 = arith.cmpf ole, %lhs, %rhs fastmath<fast> : f64
  return %4, %5, %6, %7, %8, %9, %10 : f64, f64, f64, f64, f64, f64, i1
}

// Check the default transform
// CHECK:       func.func public @foo{{.*}}() -> (f64, f64, f64, f64, f64, f64, i1) {
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:    func.return {{.*}}
// CHECK-NEXT:  }

// Check the transform with "none"
// NONE:       func.func public @foo{{.*}}() -> (f64, f64, f64, f64, f64, f64, i1) {
// NONE-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// NONE-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} : f64
// NONE-NEXT:    func.return {{.*}}
// NONE-NEXT:  }

// Check the transform with single flag
// SINGLE:       func.func public @foo{{.*}}() -> (f64, f64, f64, f64, f64, f64, i1) {
// SINGLE-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// SINGLE-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} fastmath<nnan> : f64
// SINGLE-NEXT:    func.return {{.*}}
// SINGLE-NEXT:  }

// Check the transform with two flags
// DOUBLE:       func.func public @foo{{.*}}() -> (f64, f64, f64, f64, f64, f64, i1) {
// DOUBLE-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f64, f64)
// DOUBLE-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.divf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    %{{.*}} = arith.cmpf ole, %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f64
// DOUBLE-NEXT:    func.return {{.*}}
// DOUBLE-NEXT:  }
