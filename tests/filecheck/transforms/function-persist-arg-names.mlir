// RUN: xdsl-opt %s -p function-persist-arg-names | filecheck %s

// CHECK: func.func @test(%one : i32 {llvm.name = "one"}, %two : f32 {llvm.name = "two"}, %three : f64 {llvm.name = "three"}) {
func.func @test(%one: i32, %two: f32, %three: f64) -> () {
    func.return
}


// CHECK: func.func @test2(%one : i32 {llvm.name = "preexisting_name"}, %0 : f32 {llvm.some_other_arg_attr = "some_other_val"}, %three : f64 {llvm.name = "three"}) {
func.func @test2(%one : i32 {"llvm.name" = "preexisting_name"}, %0 : f32 {"llvm.some_other_arg_attr" = "some_other_val"}, %three : f64) {
  func.return
}


// CHECK: func.func @no_arg_names(%0 : i32, %1 : f32 {llvm.some_other_arg_attr = "some_other_val"}, %2 : f64) {
func.func @no_arg_names(%0 : i32, %1 : f32 {"llvm.some_other_arg_attr" = "some_other_val"}, %2 : f64) {
  func.return
}

// CHECK: func.func @no_arg_attrs_or_names(%0 : i32, %1 : f32, %2 : f64) {
func.func @no_arg_attrs_or_names(%0 : i32, %1 : f32, %2 : f64) {
  func.return
}

// CHECK: func.func private @decl_func() -> f64
func.func private @decl_func() -> f64
