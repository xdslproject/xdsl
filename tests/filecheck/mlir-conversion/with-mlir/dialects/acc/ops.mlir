// RUN: MLIR_ROUNDTRIP

builtin.module {
  func.func @empty() {
    acc.parallel {
      acc.yield
    }
    func.return
  }
  func.func @self_if(%c : i1) {
    acc.parallel self(%c) if(%c) {
      acc.yield
    }
    func.return
  }
  func.func @combined_default_self() {
    acc.parallel combined(loop) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
  func.func @async_one_operand(%a : i64) {
    acc.parallel async(%a : i64) {
      acc.yield
    }
    func.return
  }
  func.func @num_gangs_single(%a : i32) {
    acc.parallel num_gangs(%a : i32) {
      acc.yield
    }
    func.return
  }
  func.func @num_workers_vector_length(%a : i64, %b : i32, %c : i32) {
    acc.parallel num_workers(%a : i64 [#acc.device_type<default>], %b : i32 [#acc.device_type<nvidia>]) vector_length(%c : i32) {
      acc.yield
    }
    func.return
  }
  func.func @wait_one_operand(%a : i64) {
    acc.parallel wait(%a : i64) {
      acc.yield
    }
    func.return
  }
  func.func @test_entire(%c : i1, %a : i32, %b : i64) {
    acc.parallel combined(loop) async(%b : i64) num_gangs(%a : i32) num_workers(%b : i64) vector_length(%a : i32) wait(%b : i64) self(%c) if(%c) {
      acc.yield
    } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @empty() {
// CHECK-NEXT:      acc.parallel {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @self_if(
// CHECK:           acc.parallel self(%{{.*}}) if(%{{.*}}) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK:         func.func @combined_default_self() {
// CHECK-NEXT:      acc.parallel combined(loop) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
// CHECK:         func.func @async_one_operand(
// CHECK:           acc.parallel async(%{{.*}} : i64) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK:         func.func @num_gangs_single(
// CHECK:           acc.parallel num_gangs(%{{.*}} : i32) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK:         func.func @num_workers_vector_length(
// CHECK:           acc.parallel num_workers(%{{.*}} : i64 [#acc.device_type<default>], %{{.*}} : i32 [#acc.device_type<nvidia>]) vector_length(%{{.*}} : i32) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK:         func.func @wait_one_operand(
// CHECK:           acc.parallel wait(%{{.*}} : i64) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      }
// CHECK:         func.func @test_entire(
// CHECK:           acc.parallel combined(loop) async(%{{.*}} : i64) num_gangs(%{{.*}} : i32) num_workers(%{{.*}} : i64) vector_length(%{{.*}} : i32) wait(%{{.*}} : i64) self(%{{.*}}) if(%{{.*}}) {
// CHECK-NEXT:        acc.yield
// CHECK-NEXT:      } attributes {defaultAttr = #acc<defaultvalue present>, selfAttr}
