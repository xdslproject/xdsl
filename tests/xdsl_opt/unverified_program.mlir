builtin.module {
    %a = "arith.constant"() {value = 1 : i32} : () -> i32
    %b = "arith.constant"() {value = 1.0 : f32} : () -> f32
    %c = "arith.addf"(%a, %b) : (i32, f32) -> f32
}
