"builtin.module"() ({
  %0:3 = "test.op"() : () -> (i64, index, memref<1xi64>)
  "memref.store"(%0#0, %0#2, %0#1) <{nontemporal = false}> : (i64, memref<1xi64>, index) -> ()
}) : () -> ()
