"builtin.module"() ({
  %0:3 = "test.op"() : () -> (i1, !ltl.sequence, !ltl.property)
  %1 = "ltl.and"(%0#1, %0#1) : (!ltl.sequence, !ltl.sequence) -> !ltl.sequence
  %2 = "ltl.and"(%0#2, %0#2) : (!ltl.property, !ltl.property) -> !ltl.property
  %3 = "ltl.and"(%0#0, %0#0) : (i1, i1) -> i1
}) : () -> ()
