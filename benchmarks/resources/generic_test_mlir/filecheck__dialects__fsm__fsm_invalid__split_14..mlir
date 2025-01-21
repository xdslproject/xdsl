"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : i16}> : () -> i16
  %1 = "arith.constant"() <{value = 0 : i16}> : () -> i16
  %2 = "arith.constant"() <{value = 0 : i16}> : () -> i16
  %3 = "fsm.hw_instance"(%0, %1, %2) {machine = @foo, sym_name = "foo_inst"} : (i16, i16, i16) -> i16
}) : () -> ()
