"builtin.module"() ({
  %0:3 = "test.op"() : () -> (i8, index, memref<1xi8>)
  "memref.store"(%0#0, %0#2, %0#1) <{nontemporal = false}> : (i8, memref<1xi8>, index) -> ()
}) : () -> ()
