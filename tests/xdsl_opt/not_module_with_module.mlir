%0 = "test.op"() : () -> i32
%1 = "test.op"(%0, %0) : (i32, i32) -> i32
builtin.module {
  %2 = "test.op"() : () -> i32
}
