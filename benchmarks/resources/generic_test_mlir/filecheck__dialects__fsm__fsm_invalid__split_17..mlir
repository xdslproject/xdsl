"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "qux"}> ({
    %0 = "fsm.instance"() {machine = @foo, sym_name = "foo_inst"} : () -> !fsm.instancetype
    %1 = "arith.constant"() <{value = true}> : () -> i1
    %2 = "fsm.trigger"(%1, %0) : (i1, !fsm.instancetype) -> i1
    %3 = "arith.constant"() <{value = false}> : () -> i1
    %4 = "fsm.trigger"(%3, %0) : (i1, !fsm.instancetype) -> i1
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
