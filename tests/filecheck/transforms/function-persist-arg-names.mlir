// RUN: xdsl-opt %s -p function-persist-arg-names | filecheck %s

// CHECK: func.func @test(%one : i32 {"name" = "one"}, %two : f32 {"name" = "two"}, %three : f64 {"name" = "three"}) {
func.func @test(%one: i32, %two: f32, %three: f64) -> () {
    func.return
}


//CHECK: func.func @test2(%one : i32 {"name" = "preexisting_name"}, %two : f32 {"name" = "two", "some_other_arg_attr" = "some_other_val"}, %three : f64 {"name" = "three"}) {
func.func @test2(%one : i32 {"name" = "preexisting_name"}, %two : f32 {"some_other_arg_attr" = "some_other_val"}, %three : f64) {
  func.return
}
