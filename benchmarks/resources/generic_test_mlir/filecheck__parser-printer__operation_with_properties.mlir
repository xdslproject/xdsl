"builtin.module"() ({
  "unregistered.op"() <{test = 2 : i32}> : () -> ()
  "unregistered.op"() <{test = 42 : i64, test2 = 71 : i32}> ({
  }) {test3 = "foo"} : () -> ()
  "unregistered.op"() <{test = 42 : i64}> {test = "foo"} : () -> ()
}) : () -> ()
