"builtin.module"() ({
  %0:3 = "test.op"() : () -> (i16, index, memref<1xi16>)
  "memref.store"(%0#0, %0#2, %0#1) <{nontemporal = false}> : (i16, memref<1xi16>, index) -> ()
}) : () -> ()
