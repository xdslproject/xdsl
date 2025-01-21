"builtin.module"() ({
  %0:2 = "test.op"() : () -> (!ltl.sequence, !ltl.property)
  %1 = "ltl.and"(%0#0, %0#1) : (!ltl.sequence, !ltl.property) -> !ltl.property
}) : () -> ()
